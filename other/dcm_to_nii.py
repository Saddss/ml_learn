import os
import traceback
import uuid
from typing import List

import SimpleITK as sitk


def dicom_to_nii(folder, save_dir):
    name = os.path.basename(folder)
    print(f"[{name}] Start Processing ...")
    series_ids: List[str] = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder)
    results = []
    if not series_ids:
        print(f"[{name}] No DICOM series found. Exiting.")
        return results
    print(f"[{name}] There are {len(series_ids)} series under {folder}.")
    for series_id in series_ids:
        try:
            # read each series
            series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, series_id)
            if len(series_files) < 10:
                print(f"[{name}] Series {series_id} skipped: only {len(series_files)} files.")
                continue
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_files)
            series_reader.MetaDataDictionaryArrayUpdateOn()
            series_reader.LoadPrivateTagsOn()
            image3D = series_reader.Execute()

            if not series_reader.HasMetaDataKey(0, '0020|0010'):
                study_id = 'UnknownStudyID'
            else:
                study_id = series_reader.GetMetaData(0, '0020|0010').strip().replace('-', '')

            if not series_reader.HasMetaDataKey(0, '0020|0011'):
                series_number = 'UnknownSeriesNumber'
            else:
                series_number = series_reader.GetMetaData(0, '0020|0011').strip().replace('-', '')

            os.makedirs(save_dir, exist_ok=True)
            unique_id = str(uuid.uuid4().hex[:4]).replace('-', '')
            nii_file_name = f'Study-{study_id}-Series-{series_number}-uuid-{unique_id}.nii.gz'
            save_nii_file = os.path.join(save_dir, nii_file_name)
            sitk.WriteImage(image3D, save_nii_file)
            print(f"[{name}] Series {study_id}-{series_number}-{unique_id} has been anonymized {save_nii_file}.")
        except:
            traceback_str = traceback.format_exc()
            print(f"[{name}] Processing series {series_id} error : {traceback_str}" )
            continue

    print(f"[{name}] Finish Processing!")

if __name__ == '__main__':
    dicom_to_nii('/Users/saddss/work/data/dicom/99563/', '/Users/saddss/work/data/nii/99563/')