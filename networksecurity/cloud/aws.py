import boto3
import os
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys


def syn_folder_s3(bucket_name, folder_path):
    """
    Syncs a local folder to an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_path (str): The path to the local folder to sync.

    Raises:
        NetworkSecurityException: If the folder does not exist or if there is an error during the sync process.
    """
    try:
        if not os.path.exists(folder_path):
            raise NetworkSecurityException(f"Folder {folder_path} does not exist.")

        s3 = boto3.client('s3')
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                s3.upload_file(file_path, bucket_name, file_path.replace('\\', '/'))

        logging.info(f"Successfully synced {folder_path} to S3 bucket {bucket_name}.")
    except Exception as e:
        raise NetworkSecurityException(e , sys)