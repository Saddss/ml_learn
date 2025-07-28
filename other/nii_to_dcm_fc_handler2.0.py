# -*- coding: utf-8 -*-
import oss2
import os
import json
import tempfile
import zipfile
import shutil
import logging
from datetime import datetime

import nibabel
import numpy as np
import SimpleITK as sitk
from nibabel.dft import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

# 配置日志
logger = logging.getLogger()

# class CTImageNiftiSlicePipeline:
#     @staticmethod
#     def load_nifti(nifti_path):
#         try:
#             image = sitk.ReadImage(nifti_path)
#         except Exception as e:
#             with tempfile.TemporaryDirectory() as tmp_dir:
#                 temp_file = os.path.join(tmp_dir, 'temp.nii.gz')
#                 img = nibabel.load(nifti_path)
#                 nibabel.save(nibabel.Nifti1Image(img.get_fdata(), img.get_qform()), temp_file)
#                 image = sitk.ReadImage(temp_file)
#         return image
#
#     def generate_uid(self):
#         return str(uuid.uuid4()).replace('-', '')
#
#     def nii_to_dicom(self, nifti_file, output_dir):
#         modality = 'CT'
#         try:
#             nifti_img = self.load_nifti(nifti_file)
#             pixel_array = sitk.GetArrayFromImage(nifti_img)
#
#             spacing = nifti_img.GetSpacing()
#             origin = nifti_img.GetOrigin()
#             direction = nifti_img.GetDirection()
#
#             study_date = datetime.now().strftime("%Y%m%d")
#             study_time = datetime.now().strftime("%H%M%S")
#             series_date = study_date
#             series_time = study_time
#
#             study_instance_uid = self.generate_uid()
#             series_instance_uid = self.generate_uid()
#             frame_of_reference_uid = self.generate_uid()
#             study_id = self.generate_uid()
#             patient_id = self.generate_uid()
#
#             sop_class_uid = self.get_modality_sop_class_uid(modality)
#             transfer_syntax_uid = ExplicitVRLittleEndian
#
#             os.makedirs(output_dir, exist_ok=True)
#
#             scale_slope, scale_intercept = 1.0, 0.0
#             pixel_array, pixel_range = self.prepare_pixel_data(pixel_array, modality)
#
#             for slice_idx in range(pixel_array.shape[0]):
#                 # 创建文件元信息
#                 file_meta = FileMetaDataset()
#                 file_meta.MediaStorageSOPClassUID = sop_class_uid
#                 file_meta.MediaStorageSOPInstanceUID = generate_uid()
#                 file_meta.TransferSyntaxUID = transfer_syntax_uid
#                 file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9.10"
#                 file_meta.ImplementationVersionName = "NIfTI2DICOM_v1.0"
#
#                 ds = FileDataset(f"{slice_idx}.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
#
#                 # 患者信息
#                 ds.PatientName = "xxx"
#                 ds.PatientID = patient_id
#                 ds.PatientBirthDate = ""
#                 ds.PatientSex = "O"
#                 ds.PatientAge = "000Y"
#
#                 # 研究信息
#                 ds.StudyDate = study_date
#                 ds.StudyTime = study_time
#                 ds.StudyInstanceUID = study_instance_uid
#                 ds.StudyID = study_id
#                 ds.StudyDescription = "Converted from NIfTI"
#
#                 # 系列信息
#                 ds.SeriesDate = series_date
#                 ds.SeriesTime = series_time
#                 ds.SeriesInstanceUID = series_instance_uid
#                 ds.SeriesNumber = "1"
#                 ds.Modality = modality
#                 ds.SeriesDescription = "NIfTI Conversion"
#                 ds.FrameOfReferenceUID = frame_of_reference_uid
#
#                 # 设备信息
#                 ds.Manufacturer = "NIfTI Converter"
#                 ds.InstitutionName = "Medical Imaging Lab"
#                 ds.ManufacturerModelName = "NIfTI Converter"
#                 ds.DeviceSerialNumber = "SN12345"
#                 ds.SoftwareVersions = "1.0"
#
#                 # 图像信息
#                 ds.InstanceNumber = str(slice_idx + 1)
#                 ds.ImagePositionPatient = self.calculate_slice_position(
#                     origin, direction, spacing, slice_idx, pixel_array.shape[0]
#                 )
#                 ds.ImageOrientationPatient = list(direction[:6])
#                 ds.SamplesPerPixel = 1
#                 ds.PhotometricInterpretation = "MONOCHROME2"
#                 ds.Rows = pixel_array.shape[2]
#                 ds.Columns = pixel_array.shape[1]
#                 ds.PixelSpacing = [spacing[0], spacing[1]]
#                 ds.SliceThickness = str(spacing[2])
#                 ds.SpacingBetweenSlices = str(spacing[2])
#                 ds.SliceLocation = str(slice_idx * spacing[2])
#
#                 # 像素数据属性
#                 ds.BitsAllocated = 16
#                 ds.BitsStored = 16
#                 ds.HighBit = 15
#                 ds.PixelRepresentation = 1
#                 ds.RescaleIntercept = str(scale_intercept)
#                 ds.RescaleSlope = str(scale_slope)
#                 ds.RescaleType = "HU" if modality == "CT" else ""
#
#                 # 设置窗口中心/宽度
#                 window_center = (pixel_range[0] + pixel_range[1]) / 2 * scale_slope + scale_intercept
#                 window_width = (pixel_range[1] - pixel_range[0]) * scale_slope
#                 ds.WindowCenter = str(window_center)
#                 ds.WindowWidth = str(window_width)
#
#                 # 添加备选的窗口设置
#                 if modality == "CT":
#                     ds.WindowCenterWidthExplanation = ["Soft Tissue", "Lung"]
#                     ds.add_new([0x0028, 0x1051], 'DS', "40")
#                     ds.add_new([0x0028, 0x1051], 'DS', "400")
#                     ds.add_new([0x0028, 0x1052], 'DS', "-600")
#                     ds.add_new([0x0028, 0x1052], 'DS', "1500")
#
#                 # 唯一标识符
#                 ds.SOPClassUID = sop_class_uid
#                 ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
#
#                 # 设置像素数据
#                 slice_data = pixel_array[slice_idx, :, :].astype(np.int16)
#                 ds.PixelData = slice_data.tobytes()
#
#                 filename = os.path.join(output_dir, f"IMG_{slice_idx + 1:06d}.dcm")
#                 ds.save_as(filename, write_like_original=False)
#
#             logger.info(f"成功转换 {pixel_array.shape[0]} 个切片到 {output_dir}")
#             return True
#
#         except Exception as e:
#             logger.error(f"转换失败: {str(e)}")
#             return False
#
#     @staticmethod
#     def get_modality_sop_class_uid(modality):
#         """根据模态类型返回正确的 SOP Class UID"""
#         sop_classes = {
#             "CT": "1.2.840.10008.5.1.4.1.1.2",
#             "MR": "1.2.840.10008.5.1.4.1.1.4",
#             "PT": "1.2.840.10008.5.1.4.1.1.128",
#             "CR": "1.2.840.10008.5.1.4.1.1.1",
#             "US": "1.2.840.10008.5.1.4.1.1.6.1",
#             "XA": "1.2.840.10008.5.1.4.1.1.12.1",
#             "MG": "1.2.840.10008.5.1.4.1.1.1.2",
#         }
#         return sop_classes.get(modality.upper(), "1.2.840.10008.5.1.4.1.1.2")
#
#     @staticmethod
#     def prepare_pixel_data(pixel_array, modality):
#         """准备像素数据，包括缩放和调整数据类型"""
#         min_val, max_val = np.min(pixel_array), np.max(pixel_array)
#         scale_slope, scale_intercept = 1.0, 0.0
#
#         if modality == "CT":
#             pixel_array = pixel_array.astype(np.float32)
#             scale_slope = 1.0
#             scale_intercept = -1024 if min_val < -1000 else 0.0
#         elif modality == "MR":
#             if max_val > 32767 or min_val < -32768:
#                 scale_slope = (max_val - min_val) / 65535.0
#                 scale_intercept = min_val
#                 pixel_array = ((pixel_array - min_val) / scale_slope).astype(np.int16)
#
#         if pixel_array.dtype != np.int16:
#             pixel_array = pixel_array.astype(np.int16)
#
#         return pixel_array, (min_val, max_val)
#
#     @staticmethod
#     def calculate_slice_position(origin, direction, spacing, slice_idx, num_slices):
#         """计算切片的物理位置"""
#         direction_matrix = np.array(direction).reshape(3, 3)
#         z_vector = direction_matrix[:, 2]
#         position_offset = np.array(origin) + z_vector * spacing[2] * slice_idx
#         return [position_offset[0], position_offset[1], position_offset[2]]
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

    def nii_to_dicom(self, nifti_file, output_dir):
        try:
            nifti_img = self.load_nifti(nifti_file)
            pixel_array = sitk.GetArrayFromImage(nifti_img)

            spacing = nifti_img.GetSpacing()
            origin = nifti_img.GetOrigin()
            direction = nifti_img.GetDirection()

            study_date = datetime.now().strftime('%Y%m%d')
            study_time = datetime.now().strftime('%H%M%S')
            series_date = study_date
            series_time = study_time

            study_instance_uid = pydicom.uid.generate_uid()
            series_instance_uid = pydicom.uid.generate_uid()
            frame_of_reference_uid = pydicom.uid.generate_uid()
            study_id = pydicom.uid.generate_uid()
            patient_id = pydicom.uid.generate_uid()

            transfer_syntax_uid = ExplicitVRLittleEndian

            os.makedirs(output_dir, exist_ok=True)
            scale_slope, scale_intercept = 1.0, 0.0

            for slice_idx in range(pixel_array.shape[0]):
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.TransferSyntaxUID = transfer_syntax_uid
                file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
                file_meta.ImplementationVersionName = 'NIfTI2DICOM_v1.0'

                ds = FileDataset(f'{slice_idx}.dcm', {}, file_meta=file_meta, preamble=b'\0' * 128)

                ds.PatientName = 'xxx'
                ds.PatientID = patient_id
                ds.PatientBirthDate = ''
                ds.PatientSex = 'O'
                ds.PatientAge = '000Y'

                ds.StudyDate = study_date
                ds.StudyTime = study_time
                ds.StudyInstanceUID = study_instance_uid
                ds.StudyID = study_id
                ds.StudyDescription = ''

                ds.SeriesDate = series_date
                ds.SeriesTime = series_time
                ds.SeriesInstanceUID = series_instance_uid
                ds.SeriesNumber = '1'
                ds.Modality = 'CT'
                ds.SeriesDescription = ''
                ds.FrameOfReferenceUID = frame_of_reference_uid

                ds.Manufacturer = ''
                ds.InstitutionName = ''
                ds.ManufacturerModelName = ''
                ds.DeviceSerialNumber = 'SN12345'
                ds.SoftwareVersions = '1.0'

                ds.InstanceNumber = str(slice_idx + 1)
                ds.ImagePositionPatient = self.calculate_slice_position(origin, direction, spacing, slice_idx)
                ds.ImageOrientationPatient = list(direction[:6])
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = 'MONOCHROME2'
                ds.Rows = pixel_array.shape[2]
                ds.Columns = pixel_array.shape[1]
                ds.PixelSpacing = [spacing[0], spacing[1]]
                ds.SliceThickness = str(spacing[2])
                ds.SpacingBetweenSlices = str(spacing[2])
                ds.SliceLocation = str(slice_idx * spacing[2])

                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1
                ds.RescaleIntercept = str(scale_intercept)
                ds.RescaleSlope = str(scale_slope)
                ds.RescaleType = 'HU'

                ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

                slice_data = pixel_array[slice_idx, :, :].astype(np.int16)
                ds.PixelData = slice_data.tobytes()

                filename = os.path.join(output_dir, f'IMG_{slice_idx + 1:06d}.dcm')
                ds.save_as(filename, write_like_original=False)

            print(f"Sliced completed, {pixel_array.shape[0]} images to {output_dir}")

            return True

        except Exception as e:
            print(f'Sliced error: {str(e)}')
            return False

    @staticmethod
    def calculate_slice_position(origin, direction, spacing, slice_idx):
        direction_matrix = np.array(direction).reshape(3, 3)
        z_vector = direction_matrix[:, 2]
        position_offset = np.array(origin) + z_vector * spacing[2] * slice_idx
        return [
            position_offset[0],
            position_offset[1],
            position_offset[2]
        ]

class NiiToDicomConverter:
    def __init__(self, config):
        self.config = config
        self.source = config['source']
        self.target = config['target']
        self.temp_dir = tempfile.mkdtemp(prefix='nii_to_dicom_')
        self.original_filename = None  # 存储原始文件名
        logger.info(f"创建临时工作目录: {self.temp_dir}")

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"清理临时目录: {self.temp_dir}")

    def create_oss_bucket(self, endpoint, access_key_id, access_key_secret, bucket_name):
        try:
            auth = oss2.Auth(access_key_id, access_key_secret)
            bucket = oss2.Bucket(auth, endpoint, bucket_name)
            return bucket
        except Exception as e:
            logger.error(f"创建 OSS 客户端失败: {str(e)}")
            return Exception(str(e))

    def download_from_oss(self):
        logger.info("开始从 OSS 下载 NII 文件...")

        oss_path = self.source['path']
        if not oss_path.startswith('oss://'):
            raise ValueError("源路径必须以 'oss://' 开头")

        path_parts = oss_path[6:].split('/', 1)
        bucket_name = path_parts[0]
        object_key = path_parts[1] if len(path_parts) > 1 else ''

        if not object_key:
            raise ValueError("OSS 路径格式错误，缺少对象键")

        # 提取并保存原始文件名（去掉.niiF.gz后缀）
        original_filename = os.path.basename(object_key)
        if original_filename.endswith('.nii.gz'):
            self.original_filename = original_filename[:-7]  # 去掉 .nii.gz
        elif original_filename.endswith('.nii'):
            self.original_filename = original_filename[:-4]  # 去掉 .nii
        else:
            self.original_filename = os.path.splitext(original_filename)[0]  # 去掉其他后缀

        logger.info(f"原始文件名: {original_filename}, 提取的名称: {self.original_filename}")

        bucket = self.create_oss_bucket(
            self.source['endpoint'],
            self.source['ak'],
            self.source['sk'],
            bucket_name
        )

        if not bucket:
            raise Exception("无法创建 OSS 客户端")

        local_nii_path = os.path.join(self.temp_dir, os.path.basename(object_key))

        try:
            bucket.get_object_to_file(object_key, local_nii_path)
            logger.info(f"下载完成: {oss_path} -> {local_nii_path}")
            return local_nii_path
        except Exception as e:
            raise Exception(str(e))

    def convert_nii_to_dicom(self, nii_file):
        logger.info("开始 NII 到 DICOM 转换...")

        dicom_output_dir = os.path.join(self.temp_dir, 'dicom_output')
        os.makedirs(dicom_output_dir, exist_ok=True)

        converter = CTImageNiftiSlicePipeline()
        success = converter.nii_to_dicom(nii_file, dicom_output_dir)

        if not success:
            raise Exception("NII 到 DICOM 转换失败")

        logger.info(f"转换完成，输出目录: {dicom_output_dir}")
        return dicom_output_dir

    def zip_directory(self, source_dir, zip_path):
        logger.info(f"开始打包目录: {source_dir}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arc_name)

        logger.info(f"打包完成: {zip_path}")
        return zip_path

    def upload_to_oss(self, local_path, is_zip=False):
        logger.info("开始上传到目标 OSS...")

        target_path = self.target['path']
        if target_path.startswith('oss://'):
            path_parts = target_path[6:].split('/', 1)
            bucket_name = path_parts[0]
            base_path = path_parts[1] if len(path_parts) > 1 else ''
        else:
            bucket_name = target_path
            base_path = ''

        base_path = base_path.rstrip('/')

        bucket = self.create_oss_bucket(
            self.target['endpoint'],
            self.target['ak'],
            self.target['sk'],
            bucket_name
        )

        if not bucket:
            raise Exception("无法创建目标 OSS 客户端")

        try:
            # 使用原始文件名而不是时间戳
            output_name = self.original_filename if self.original_filename else 'dicom_output'

            if is_zip:
                object_key = f"{base_path}/{output_name}.zip" if base_path else f"{output_name}.zip"
                object_key = object_key.lstrip('/')
                bucket.put_object_from_file(object_key, local_path)
                logger.info(f"ZIP 文件上传完成: {object_key}")
                # 返回完整的 OSS 路径
                full_path = f"oss://{bucket_name}/{object_key}"
                return full_path
            else:
                base_object_path = f"{base_path}/{output_name}" if base_path else output_name
                base_object_path = base_object_path.lstrip('/')

                uploaded_files = []
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, local_path)
                        object_key = f"{base_object_path}/{relative_path}".replace('\\', '/')
                        bucket.put_object_from_file(object_key, file_path)
                        uploaded_files.append(object_key)

                logger.info(f"目录上传完成，共上传 {len(uploaded_files)} 个文件到 {base_object_path}/")
                # 返回完整的 OSS 路径
                full_path = f"oss://{bucket_name}/{base_object_path}/"
                return full_path

        except Exception as e:
            raise Exception(str(e))

    def process(self):
        try:
            logger.info("NII 到 DICOM 转换流程开始")

            nii_file = self.download_from_oss()
            dicom_dir = self.convert_nii_to_dicom(nii_file)

            if self.target.get('type', '').lower() == 'local':
                zip_filename = f"{self.original_filename}.zip" if self.original_filename else 'dicom_output.zip'
                zip_path = os.path.join(self.temp_dir, zip_filename)
                self.zip_directory(dicom_dir, zip_path)
                result_path = self.upload_to_oss(zip_path, is_zip=True)
            else:
                result_path = self.upload_to_oss(dicom_dir, is_zip=False)
            return result_path
        except Exception as e:
            raise Exception(str(e))

def handler(event, context):
    logger.info(f"receive event: {event}")
    try:
        eventJson = json.loads(event)
        converter = NiiToDicomConverter(eventJson)
        result = converter.process()
        # 修改返回格式为指定格式
        return {
            "success": True,
            "msg": "",
            "data": result
        }
    except Exception as e:
        # 错误时data为None，msg为异常信息
        return {
            "success": False,
            "msg": str(e),
            "data": None
        }