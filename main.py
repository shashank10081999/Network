import os
import sys
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
import json
from dotenv import load_dotenv
import pandas as pd
from pymongo import MongoClient
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
from networksecurity.contants import training_pipeline
from networksecurity.compoments.data_ingestion import DataIngestion
from networksecurity.compoments.data_validation import DataValidation
from networksecurity.compoments.model_trainer import ModelTrainer
from networksecurity.compoments.data_transformation import DataTransformation
from networksecurity.entity.Config_Entity import TrainingPipelineConfig , DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact , ModelTrainerArtifact

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        logging.info("Data ingestion completed successfully.")
        logging.info(f"Train file path: {data_ingestion_artifact.train_file_path}")
        logging.info(f"Test file path: {data_ingestion_artifact.test_file_path}")

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        
        logging.info("Data validation completed successfully.")

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.intiate_date_transformation()

        logging.info("Data transformation completed successfully.")

        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)