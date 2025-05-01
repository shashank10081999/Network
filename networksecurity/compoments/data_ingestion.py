from networksecurity.contants import training_pipeline
from networksecurity.entity.Config_Entity import TrainingPipelineConfig , DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from sklearn.model_selection import train_test_split
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
import json

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def export_collection_as_dataframe(self):
        try:
            client = MongoClient(os.getenv("MONGO_DB_URL"))
            db = client[self.data_ingestion_config.data_ingestion_database_name]
            collection = db[self.data_ingestion_config.data_ingestion_collection_name]
            data = pd.DataFrame(list(collection.find()))
            data.drop(columns=["_id"], inplace=True, axis=1)
            return data
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_data_into_feature_store(self, data: pd.DataFrame):
        try:
            os.makedirs(self.data_ingestion_config.feature_store_dir, exist_ok=True)
            data.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info(f"Data exported to feature store at {self.data_ingestion_config.feature_store_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def split_data_as_train_test(self, data: pd.DataFrame):
        try:
            train_set , test_set = train_test_split(data, test_size=self.data_ingestion_config.data_ingestion_train_test_split_ration, random_state=42)
            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False)
            
            logging.info(f"Train and test data exported to {self.data_ingestion_config.train_file_path} and {self.data_ingestion_config.test_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            data = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(data)
            self.split_data_as_train_test(data)
            return DataIngestionArtifact(self.data_ingestion_config.train_file_path , self.data_ingestion_config.test_file_path)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)