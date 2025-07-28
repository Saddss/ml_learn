# -*- coding: utf-8 -*-
import logging

import oss2
import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import threading
import json
import pydicom
from pydicom.uid import CTImageStorage
import uuid
import shutil
import concurrent.futures
import random
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits  # 包含字母和数字
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

accessKeyId = os.environ["oss_access_key_id"]
accessKeySecret = os.environ["oss_access_key_secret"]
endpoint = os.environ["oss_endpoint"]
bucketName = os.environ["oss_bucket_name"]


auth = oss2.Auth(accessKeyId, accessKeySecret)
logger = logging.getLogger()
# 创建文件处理器
file_handler = logging.FileHandler('myLogfile.log')  # 指定日志文件路径
file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别

# 将处理器添加到 logger
logger.addHandler(file_handler)


def upload_directory(bucket, local_directory, oss_prefix):
    try:

        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                # 计算 OSS 路径
                fileName = os.path.basename(local_path)
                oss_path = os.path.join(oss_prefix, fileName)

                # 上传文件到 OSS
                bucket.put_object_from_file(oss_path, local_path)
                # logger.info("uploading %s to oss %s success", local_path, oss_path)
        dcm_list = [f for f in os.listdir(local_directory) if f.endswith('.dcm')]
        dcm_list = sorted(dcm_list)
        dcm_list_file = 'dcm_list.json'
        with open(dcm_list_file, 'w', encoding='utf-8') as f:
            json.dump(dcm_list, f, ensure_ascii=False, indent=4)
        bucket.put_object_from_file(os.path.join(oss_prefix, dcm_list_file), dcm_list_file)
        os.remove(dcm_list_file)

    except Exception as e:
        logger.info(e)
        logger.info("upload to %s failed", oss_prefix)
        raise ValueError("执行失败")

# 切分nii文件为dcm
def nii_to_dicom(nifti_file, dicom_output_directory):
    try:
        # 读取 NIfTI 文件
        image = sitk.ReadImage(nifti_file)

        # 创建一个 DICOM image series writer
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = np.array(image.GetDirection()).reshape(3, 3)

        # 如果不存在，则创建输出目录
        if not os.path.exists(dicom_output_directory):
            os.makedirs(dicom_output_directory)

        # 对于图像的每一层切片，将其单独写为一个 DICOM 文件
        for i in range(image.GetDepth()):
            # 提取单个切片
            slice_i = image[:, :, i]

            # 创建一些基本的 DICOM 标签作为元数据
            # 这些通常在实际使用时需要从实际数据或用户输入中获得
            # None 是用来表示可选的值，通常需要替换为实际值
            slice_i.SetMetaData("0008|0012", "20231001")  # Instance Creation Date
            slice_i.SetMetaData("0008|0013", "133000")   # Instance Creation Time
            slice_i.SetMetaData("0008|0060", "MR")       # Modality
            slice_i.SetMetaData("0010|0010", "PatientName")  # PatientsName
            slice_i.SetMetaData("0020|000e", str(i))     # Series Instance UID, unique for each series
            slice_i.SetMetaData("0020|1041", str(i))     # Slice Location
            slice_i.SetMetaData('0020|0011', str(i + 1))

            # 计算每个切片的物理坐标
            image_position_patient = origin + i * spacing[2] * direction[:, 2]
            position_x = image_position_patient[0]
            position_y = image_position_patient[1]
            position_z = image_position_patient[2]
            position_str = f"{position_x}\\{position_y}\\{position_z}"
            slice_i.SetMetaData("0020|0032", position_str)

            # 生成每一层切片的文件名
            filename = os.path.join(dicom_output_directory, f"slice_{i+1:03d}.dcm")

            # 写入 DICOM 文件
            writer.SetFileName(filename)
            writer.Execute(slice_i)

            # 设置meta信息
            # 读取现有的 DICOM 文件
            ds = pydicom.dcmread(filename)

            # 提取原始像素数据
            pixel_array = ds.pixel_array

            # 将像素数据转换为int16
            pixel_array_int16 = pixel_array.astype(np.int16)

            # 更新dataset中的PixelData
            ds.PixelData = pixel_array_int16.tobytes()

            # 根据需要更新其他相关的元数据，如BitsAllocated, PixelRepresentation等
            ds.BitsAllocated = 16
            ds.PixelRepresentation = 1  # 1表示有符号整数

            # 设置 SOP Class UID 为 CT Image Storage
            ds.SOPClassUID = CTImageStorage
            # ds.MediaStorageSOPClassUID = CTImageStorage
            ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            ds.Modality = 'CT'

            # 保存更改到新的 DICOM 文件
            ds.save_as(filename)
    except Exception as e:
        logger.info(e)
        logger.info("slice %s failed", nifti_file)
        raise ValueError("执行失败")


# 提取mask标签
def extract_mask_label(itemUid, maskPath, bucket):
    # 处理mask
    if maskPath is not None and maskPath.endswith('.nii.gz'):
        mask_fileKey = maskPath.replace("oss://damo-data-annotation-public/", "")
        uuid_64_prefix = str(uuid.uuid4()).replace('-', '')[:16]
        tmpMaskLocalFile = uuid_64_prefix + '_' + os.path.basename(mask_fileKey)
        try:
            bucket.get_object_to_file(mask_fileKey, tmpMaskLocalFile)
            img = nib.load(tmpMaskLocalFile)
            data = img.get_fdata()
            unique_labels = np.unique(data)
            unique_labels = unique_labels[unique_labels != 0].astype(int)
            print("提取出的所有标签值：", unique_labels)
            json_str = json.dumps(unique_labels.tolist(), ensure_ascii=False)
            dcmOssDir = f'{os.path.dirname(mask_fileKey)}/{itemUid}'
            bucket.put_object(os.path.join(dcmOssDir, 'mask_labels.txt'), json_str.encode('utf-8'))
        finally:
            if os.path.exists(tmpMaskLocalFile):
                os.remove(tmpMaskLocalFile)


def process_sub(sub):
    thread_id = threading.get_ident()
    logger.info("start to process sub length: %s with thread id: %s", len(sub), thread_id)
    bucket = oss2.Bucket(auth, endpoint, bucketName, enable_crc=False)
    try:
        for dataRecord in sub:
            itemUid = dataRecord['id']
            ossPath = dataRecord['path']
            maskPath = dataRecord.get('maskPath', None)
            if ossPath.endswith('.nii.gz'):
                fileKey = ossPath.replace("oss://damo-data-annotation-public/", "")
                dcmOssDir = f'{os.path.dirname(fileKey)}/{itemUid}'
                uuid_64_prefix = str(uuid.uuid4()).replace('-', '')[:16]
                tmpLocalFile = uuid_64_prefix + '_' + os.path.basename(fileKey)
                tmpLocalDir = str(uuid.uuid4())
                try:
                    bucket.get_object_to_file(fileKey, tmpLocalFile)
                    nii_to_dicom(tmpLocalFile, tmpLocalDir)
                    upload_directory(bucket, tmpLocalDir, dcmOssDir)
                    extract_mask_label(itemUid, maskPath, bucket)
                finally:
                    if os.path.exists(tmpLocalFile):
                        os.remove(tmpLocalFile)
                    if os.path.exists(tmpLocalDir):
                        shutil.rmtree(tmpLocalDir)
            logger.info(f"processed sub len:{len(sub)} with thread id: {thread_id}")
    except Exception as e:
        logger.info(e)
        logger.info("slice sub len %s failed", len(sub))
        raise ValueError("执行失败")
    return 'true'


def split_list(data, num_sublists):
    avg = len(data) // num_sublists
    remainder = len(data) % num_sublists
    result = []
    start = 0
    for i in range(num_sublists):
        extra = 1 if i < remainder else 0
        end = start + avg + extra
        result.append(data[start:end])
        start = end
    return result


def handler(event, context):
    try:
        logger.info("receive event: %s", event)
        bucket = oss2.Bucket(auth, endpoint, bucketName, enable_crc=False)
        eventJson = json.loads(event)

        data_arr = []
        metaFile = eventJson.get('metaFile')
        if metaFile is not None:
            tmpLocalMetaFile = 'metaFile_' + str(generate_random_string(10)) + '.jsonl'
            if os.path.exists(tmpLocalMetaFile):
                os.remove(tmpLocalMetaFile)
            bucket.get_object_to_file(metaFile.replace("oss://damo-data-annotation-public/", ""), tmpLocalMetaFile)
            with open(tmpLocalMetaFile, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            data_arr.append(data)
                        except json.JSONDecodeError as e:
                            print(f"parse line to json error: {e}，error line content：{line}")
            os.remove(tmpLocalMetaFile)
        else:
            subItem = eventJson.get('itemList')
            if subItem is not None:
                data_arr = subItem

        if data_arr is None or len(data_arr) == 0:
            logger.info("no data to process")
            print("no data to process")
            return

        num_threads = 8
        if len(data_arr) < num_threads:
            num_threads = len(data_arr)

        sub_list = split_list(data_arr, num_threads)
        # 将输入文件列表切分为多个子列表
        logger.info("total subfile num: %s", str(len(sub_list)))

        from concurrent.futures import ProcessPoolExecutor
        print(f"Using {num_threads} workers.")

        # 创建进程池
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            # 创建一个任务列表
            futures = []
            for sub in sub_list:
                futures.append(executor.submit(process_sub, sub))
            # 获取结果
            for future in concurrent.futures.as_completed(futures):
                logger.info("future.result: %s", future.result())
        logger.info("process success: %s", eventJson)
    except Exception as e:
        logger.info(e)
        raise ValueError("执行失败")
