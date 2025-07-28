# -*- coding: utf-8 -*-
import os
import json
import tempfile
import uuid

import nibabel
import numpy as np
import SimpleITK as sitk
from pydicom.uid import generate_uid
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from PIL import Image
from datetime import datetime


class CTImageNiftiSlicePipeline:

    @staticmethod
    def load_nifti(nifti_path):
        try:
            image = sitk.ReadImage(nifti_path)
        except Exception as e:
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_file = os.path.join(tmp_dir, 'temp.nii.gz')
                img = nibabel.load(nifti_path)
                nibabel.save(nibabel.Nifti1Image(img.get_fdata(), img.get_qform()), temp_file)
                image = sitk.ReadImage(temp_file)
        return image

    def generate_uid(self):
        return str(uuid.uuid4()).replace('-', '')

    def nii_to_dicom(self, nifti_file, output_dir):
        modality = 'CT'
        try:
            nifti_img = self.load_nifti(nifti_file)
            pixel_array = sitk.GetArrayFromImage(nifti_img)

            spacing = nifti_img.GetSpacing()
            origin = nifti_img.GetOrigin()
            direction = nifti_img.GetDirection()

            study_date = datetime.now().strftime("%Y%m%d")
            study_time = datetime.now().strftime("%H%M%S")
            series_date = study_date
            series_time = study_time

            study_instance_uid = self.generate_uid()
            series_instance_uid = self.generate_uid()
            frame_of_reference_uid = self.generate_uid()
            study_id = self.generate_uid()
            patient_id = self.generate_uid()

            sop_class_uid = self.get_modality_sop_class_uid(modality)
            transfer_syntax_uid = ExplicitVRLittleEndian

            os.makedirs(output_dir, exist_ok=True)

            scale_slope, scale_intercept = 1.0, 0.0
            pixel_array, pixel_range = self.prepare_pixel_data(pixel_array, modality)

            for slice_idx in range(pixel_array.shape[0]):
                # 创建文件元信息
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = sop_class_uid
                file_meta.MediaStorageSOPInstanceUID = generate_uid()
                file_meta.TransferSyntaxUID = transfer_syntax_uid
                file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9.10"  # 自定义实现 UID
                file_meta.ImplementationVersionName = "NIfTI2DICOM_v1.0"

                ds = FileDataset(f"{slice_idx}.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)

                # =================== 患者信息 ===================
                ds.PatientName = "xxx"
                ds.PatientID = patient_id
                ds.PatientBirthDate = ""  # 可选
                ds.PatientSex = "O"  # O=其他, M=男, F=女
                ds.PatientAge = "000Y"  # 格式: NNNY, NNNM, NNNW

                # =================== 研究信息 ===================
                ds.StudyDate = study_date
                ds.StudyTime = study_time
                ds.StudyInstanceUID = study_instance_uid
                ds.StudyID = study_id
                ds.StudyDescription = "Converted from NIfTI"

                # =================== 系列信息 ===================
                ds.SeriesDate = series_date
                ds.SeriesTime = series_time
                ds.SeriesInstanceUID = series_instance_uid
                ds.SeriesNumber = "1"
                ds.Modality = modality
                ds.SeriesDescription = "NIfTI Conversion"
                ds.FrameOfReferenceUID = frame_of_reference_uid

                # =================== 设备信息 ===================
                ds.Manufacturer = "NIfTI Converter"
                ds.InstitutionName = "Medical Imaging Lab"
                ds.ManufacturerModelName = "NIfTI Converter"
                ds.DeviceSerialNumber = "SN12345"
                ds.SoftwareVersions = "1.0"

                # =================== 图像信息 ===================
                ds.InstanceNumber = str(slice_idx + 1)
                ds.ImagePositionPatient = self.calculate_slice_position(
                    origin, direction, spacing, slice_idx, pixel_array.shape[0]
                )
                ds.ImageOrientationPatient = list(direction[:6])
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.Rows = pixel_array.shape[2]
                ds.Columns = pixel_array.shape[1]
                ds.PixelSpacing = [spacing[0], spacing[1]]
                ds.SliceThickness = str(spacing[2])
                ds.SpacingBetweenSlices = str(spacing[2])
                ds.SliceLocation = str(slice_idx * spacing[2])

                # =================== 像素数据属性 ===================
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1  # 1=有符号整数, 0=无符号整数
                ds.RescaleIntercept = str(scale_intercept)
                ds.RescaleSlope = str(scale_slope)
                ds.RescaleType = "HU" if modality == "CT" else ""

                # 设置窗口中心/宽度 (基于像素值范围)
                window_center = (pixel_range[0] + pixel_range[1]) / 2 * scale_slope + scale_intercept
                window_width = (pixel_range[1] - pixel_range[0]) * scale_slope
                ds.WindowCenter = str(window_center)
                ds.WindowWidth = str(window_width)

                # 添加备选的窗口设置
                if modality == "CT":
                    ds.WindowCenterWidthExplanation = ["Soft Tissue", "Lung"]
                    ds.add_new([0x0028, 0x1051], 'DS', "40")  # 软组织窗中心
                    ds.add_new([0x0028, 0x1051], 'DS', "400")  # 软组织窗宽
                    ds.add_new([0x0028, 0x1052], 'DS', "-600")  # 肺窗中心
                    ds.add_new([0x0028, 0x1052], 'DS', "1500")  # 肺窗宽

                # =================== 唯一标识符 ===================
                ds.SOPClassUID = sop_class_uid
                ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

                # =================== 设置像素数据 ===================
                slice_data = pixel_array[slice_idx, :, :].astype(np.int16)
                ds.PixelData = slice_data.tobytes()

                # 添加序列参考 (可选)
                # ds.ReferencedSeriesSequence = Sequence()
                # ref_series = Dataset()
                # ref_series.SeriesInstanceUID = series_instance_uid
                # ds.ReferencedSeriesSequence.append(ref_series)

                filename = os.path.join(output_dir, f"IMG_{slice_idx + 1:06d}.dcm")
                ds.save_as(filename, write_like_original=False)

            print(f"成功转换 {pixel_array.shape[0]} 个切片到 {output_dir}")
            return True

        except Exception as e:
            print(f"转换失败: {str(e)}")
            return False

    @staticmethod
    def get_modality_sop_class_uid(modality):
        """根据模态类型返回正确的 SOP Class UID"""
        sop_classes = {
            "CT": "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
            "MR": "1.2.840.10008.5.1.4.1.1.4",  # MR Image Storage
            "PT": "1.2.840.10008.5.1.4.1.1.128",  # Positron Emission Tomography Image Storage
            "CR": "1.2.840.10008.5.1.4.1.1.1",  # Computed Radiography Image Storage
            "US": "1.2.840.10008.5.1.4.1.1.6.1",  # Ultrasound Image Storage
            "XA": "1.2.840.10008.5.1.4.1.1.12.1",  # X-Ray Angiographic Image Storage
            "MG": "1.2.840.10008.5.1.4.1.1.1.2",  # Mammography Image Storage
        }
        return sop_classes.get(modality.upper(), "1.2.840.10008.5.1.4.1.1.2")  # 默认为 CT

    @staticmethod
    def prepare_pixel_data(pixel_array, modality):
        """准备像素数据，包括缩放和调整数据类型"""
        min_val, max_val = np.min(pixel_array), np.max(pixel_array)
        scale_slope, scale_intercept = 1.0, 0.0

        # 根据模态类型进行特定处理
        if modality == "CT":
            # CT 值通常需要保留负值
            pixel_array = pixel_array.astype(np.float32)
            scale_slope = 1.0
            scale_intercept = -1024 if min_val < -1000 else 0.0

        elif modality == "MR":
            # MR 值通常为正，但范围可能很大
            if max_val > 32767 or min_val < -32768:
                scale_slope = (max_val - min_val) / 65535.0
                scale_intercept = min_val
                pixel_array = ((pixel_array - min_val) / scale_slope).astype(np.int16)

        # 转换为16位整数
        if pixel_array.dtype != np.int16:
            pixel_array = pixel_array.astype(np.int16)

        return pixel_array, (min_val, max_val)

    @staticmethod
    def calculate_slice_position(origin, direction, spacing, slice_idx, num_slices):
        """计算切片的物理位置"""
        # 将方向向量转换为3x3矩阵
        direction_matrix = np.array(direction).reshape(3, 3)

        # 计算z轴方向向量
        z_vector = direction_matrix[:, 2]

        # 计算位置偏移 (考虑图像原点)
        position_offset = np.array(origin) + z_vector * spacing[2] * slice_idx

        # DICOM要求位置信息格式为[x, y, z]
        return [
            position_offset[0],
            position_offset[1],
            position_offset[2]
        ]

    @staticmethod
    def save_metadata_json(ds, json_path):
        """将DICOM元数据保存为JSON文件"""
        try:
            metadata = {}

            # 遍历所有DICOM元素
            for elem in ds:
                # 跳过像素数据和大对象
                if elem.tag == (0x7fe0, 0x0010) or elem.VR in ['OB', 'OW', 'UN']:
                    continue

                # 处理序列数据
                if elem.VR == 'SQ':
                    sequences = []
                    for seq_item in elem.value:
                        seq_dict = {}
                        for seq_elem in seq_item:
                            if seq_elem.VR not in ['OB', 'OW', 'UN']:
                                seq_dict[seq_elem.name] = str(seq_elem.value)
                        sequences.append(seq_dict)
                    metadata[elem.name] = sequences
                else:
                    metadata[elem.name] = str(elem.value)

            # 保存到JSON文件
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"无法保存元数据JSON: {str(e)}")


if __name__ == '__main__':
    dcm_to_nii_fc_handler = CTImageNiftiSlicePipeline()
    dcm_to_nii_fc_handler.nii_to_dicom('/Users/saddss/work/data/nii/99563/test.nii.gz', '/Users/saddss/work/data/dicom/test/')