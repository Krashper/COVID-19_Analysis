from minio import Minio
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import logging


def save_data_to_s3(bucket_name: str, file_name: str):
    try:
        hook = S3Hook(aws_conn_id="minio_conn")

        hook.load_file(
            filename=f'dags/data/{file_name}',
            key=f'datasets/{file_name}',  # указываем путь к папке
            bucket_name=bucket_name,
            replace=True
        )
    
    except Exception as e:
        logging.error("Ошибка при добавлении файла в S3 bucket: ", e)
    