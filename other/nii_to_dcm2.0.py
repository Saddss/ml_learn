# -*- coding: utf-8 -*-
import os
import tempfile

import nibabel
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
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