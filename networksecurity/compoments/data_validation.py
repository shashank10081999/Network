from networksecurity.contants import training_pipeline
from networksecurity.contants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.entity.Config_Entity import TrainingPipelineConfig , DataIngestionConfig , DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from dotenv import load_dotenv
load_dotenv()
import json
from networksecurity.utils.utils import read_yaml_file, write_yaml_file
import pandas as pd
from scipy.stats import ks_2samp

class DataValidation():
    def __init__(self , data_ingestion_artifact:DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.scghema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def read_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            expected_columns_number = len(self.scghema_config["columns"])
            actual_columns_number = df.shape[1]
            if expected_columns_number == actual_columns_number:
                return True
            else:
                return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def detech_dataset_dift(self , base_df , current_df , threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report[column] = {
                    "pvalue": float(is_same_dist.pvalue),
                    "same_distribution": is_found
                }
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)

            write_yaml_file(self.data_validation_config.drift_report_file_path, report , replace=False)

            return status
                 
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self) -> DataIngestionArtifact:
        try:
            logging.info("Data Validation started")
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            if not self.validate_number_of_columns(train_df):
                print("Number of columns in the training data does not match the expected number of columns.")
            
            if not self.validate_number_of_columns(test_df):
                print("Number of columns in the testing data does not match the expected number of columns.")
            
            if not os.path.exists(self.data_validation_config.validated_dir):
                os.makedirs(self.data_validation_config.validated_dir, exist_ok=True)
            
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            validation_status = self.detech_dataset_dift(train_df , test_df)

            return DataValidationArtifact(
                validation_status = validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        