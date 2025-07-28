# -*- coding: utf-8 -*-

import oss2
import os
import json
import tempfile
import zipfile
import shutil
import logging
import uuid
import traceback
from typing import List

import SimpleITK as sitk

# 配置日志
logger = logging.getLogger()

class DicomToNiiConverter:
    def __init__(self, config):
        self.config = config
        self.source = config['source']
        self.target = config['target']
        self.temp_dir = tempfile.mkdtemp(prefix='dicom_to_nii_')
        self.original_dirname = None  # 存储原始目录名
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
            logger.error(str(e))
            raise Exception(str(e))

    def download_from_oss(self):
        logger.info("开始从 OSS 下载 DICOM 文件...")

        oss_path = self.source['path']
        if not oss_path.startswith('oss://'):
            raise ValueError("源路径必须以 'oss://' 开头")

        path_parts = oss_path[6:].split('/', 1)
        bucket_name = path_parts[0]
        object_key = path_parts[1] if len(path_parts) > 1 else ''

        if not object_key:
            raise ValueError("OSS 路径格式错误，缺少对象键")

        bucket = self.create_oss_bucket(
            self.source['endpoint'],
            self.source['ak'],
            self.source['sk'],
            bucket_name
        )

        if not bucket:
            raise Exception("无法创建 OSS 客户端")

        # 下载整个DICOM目录
        local_dicom_dir = os.path.join(self.temp_dir, 'dicom_input')
        os.makedirs(local_dicom_dir, exist_ok=True)

        # 提取目录名作为原始目录名
        self.original_dirname = os.path.basename(object_key.rstrip('/'))

        try:
            # 列举并下载目录中的所有文件 - 修复分页问题
            downloaded_count = 0
            continuation_token = None
            total_files_found = 0

            # 首先统计总文件数
            logger.info("正在统计 OSS 目录中的 .dcm 文件总数...")
            count_token = None
            while True:
                if count_token:
                    count_result = bucket.list_objects_v2(prefix=object_key, continuation_token=count_token)
                else:
                    count_result = bucket.list_objects_v2(prefix=object_key)

                for obj in count_result.object_list:
                    # 只统计.dcm文件
                    if not obj.key.endswith('/') and obj.key.lower().endswith('.dcm'):
                        total_files_found += 1

                if count_result.is_truncated:
                    count_token = count_result.next_continuation_token
                else:
                    break

            logger.info(f"OSS 目录中共发现 {total_files_found} 个 .dcm 文件")

            # 使用分页处理下载所有文件
            while True:
                if continuation_token:
                    objects = bucket.list_objects_v2(prefix=object_key, continuation_token=continuation_token)
                else:
                    objects = bucket.list_objects_v2(prefix=object_key)

                for obj in objects.object_list:
                    if obj.key.endswith('/'):
                        continue  # 跳过目录

                    # 只下载.dcm文件
                    if not obj.key.lower().endswith('.dcm'):
                        continue  # 跳过非.dcm文件

                    relative_path = os.path.relpath(obj.key, object_key)
                    local_file_path = os.path.join(local_dicom_dir, relative_path)

                    # 创建必要的目录
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    bucket.get_object_to_file(obj.key, local_file_path)
                    downloaded_count += 1

                    # 每下载50个文件输出一次进度
                    if downloaded_count % 50 == 0:
                        logger.info(f"已下载 {downloaded_count}/{total_files_found} 个 .dcm 文件...")

                # 检查是否还有更多页面
                if objects.is_truncated:
                    continuation_token = objects.next_continuation_token
                else:
                    break

            logger.info(f"目录下载完成，共下载 {downloaded_count} 个 .dcm 文件到: {local_dicom_dir}")

            # 验证下载的文件数量
            if downloaded_count != total_files_found:
                logger.warning(f"下载文件数量不匹配：预期 {total_files_found} 个，实际下载 {downloaded_count} 个")
            else:
                logger.info(f"文件下载验证成功：{downloaded_count} 个 .dcm 文件全部下载完成")

            return local_dicom_dir

        except Exception as e:
            raise Exception(str(e))

    def dicom_to_nii(self, dicom_folder, save_dir):
        name = os.path.basename(dicom_folder)
        logger.info(f"[{name}] 开始处理 DICOM 转换...")

        # 统计本地DICOM文件数量
        total_dicom_files = 0
        for root, dirs, files in os.walk(dicom_folder):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')) or not '.' in file:
                    total_dicom_files += 1
        logger.info(f"[{name}] 本地DICOM目录中共有 {total_dicom_files} 个文件")

        series_ids: List[str] = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_folder)
        results = []

        if not series_ids:
            logger.warning(f"[{name}] 未发现 DICOM 系列文件")
            return results

        logger.info(f"[{name}] 发现 {len(series_ids)} 个系列文件")
        os.makedirs(save_dir, exist_ok=True)

        processed_files_count = 0
        for series_id in series_ids:
            try:
                series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder, series_id)
                logger.info(f"[{name}] 系列 {series_id} 包含 {len(series_files)} 个文件")

                # 降低文件数量限制，避免跳过有效的小系列
                if len(series_files) < 5:
                    logger.info(f"[{name}] 系列 {series_id} 跳过: 仅有 {len(series_files)} 个文件 (< 5)")
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

                unique_id = str(uuid.uuid4().hex[:4]).replace('-', '')
                nii_file_name = f'Study-{study_id}-Series-{series_number}-uuid-{unique_id}.nii.gz'
                save_nii_file = os.path.join(save_dir, nii_file_name)

                sitk.WriteImage(image3D, save_nii_file)
                results.append(save_nii_file)
                processed_files_count += len(series_files)
                logger.info(f"[{name}] 系列 {study_id}-{series_number}-{unique_id} 已转换: {save_nii_file} (使用了 {len(series_files)} 个文件)")

            except Exception as e:
                traceback_str = traceback.format_exc()
                logger.error(f"[{name}] 处理系列 {series_id} 时出错: {traceback_str}")
                continue

        logger.info(f"[{name}] 转换完成! 共生成 {len(results)} 个 NII 文件，处理了 {processed_files_count} 个 DICOM 文件")

        if processed_files_count < total_dicom_files:
            logger.warning(f"[{name}] 注意：总共有 {total_dicom_files} 个 DICOM 文件，但只处理了 {processed_files_count} 个")

        return results

    def convert_dicom_to_nii(self, dicom_dir):
        logger.info("开始 DICOM 到 NII 转换...")

        nii_output_dir = os.path.join(self.temp_dir, 'nii_output')
        os.makedirs(nii_output_dir, exist_ok=True)

        results = self.dicom_to_nii(dicom_dir, nii_output_dir)

        if not results:
            raise Exception("DICOM 到 NII 转换失败：未生成任何 NII 文件")

        logger.info(f"转换完成，输出目录: {nii_output_dir}")
        return nii_output_dir

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
            # 使用原始目录名
            output_name = self.original_dirname if self.original_dirname else 'nii_output'

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
            logger.info("DICOM 到 NII 转换流程开始")

            dicom_dir = self.download_from_oss()
            nii_dir = self.convert_dicom_to_nii(dicom_dir)

            if self.target.get('type', '').lower() == 'local':
                # 本地下载：打包成ZIP文件
                zip_filename = f"{self.original_dirname}.zip" if self.original_dirname else 'nii_output.zip'
                zip_path = os.path.join(self.temp_dir, zip_filename)
                self.zip_directory(nii_dir, zip_path)
                result_path = self.upload_to_oss(zip_path, is_zip=True)
                logger.info(f"本地下载模式：已打包成ZIP文件并上传到 {result_path}")

                # 修改返回格式为指定格式
                return result_path
            else:
                # OSS下载：直接上传NII文件
                result_path = self.upload_to_oss(nii_dir, is_zip=False)
                logger.info(f"OSS下载模式：已直接上传NII文件到 {result_path}")

                # 修改返回格式为指定格式
                return result_path

        except Exception as e:
            logger.error(str(e))
            # 错误时data为None，msg为异常信息
            raise Exception(str(e))
def handler(event, context):
    try:
        logger.info(f"receive event: {event}")

        # 解析事件数据
        eventJson = json.loads(event)
        converter = DicomToNiiConverter(eventJson)
        result = converter.process()
        # 修改返回格式为指定格式
        return {
            "success": True,
            "msg": "",
            "data": result
        }
    except Exception as e:
        logger.error(str(e))
        # 错误时data为None，msg为异常信息
        return {
            "success": False,
            "msg": str(e),
            "data": None
        }