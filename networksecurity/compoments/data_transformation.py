from networksecurity.contants import training_pipeline
from networksecurity.entity.Config_Entity import TrainingPipelineConfig , DataIngestionConfig , DataValidationConfig , DataTransformationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from dotenv import load_dotenv
load_dotenv()
import json
from networksecurity.utils.utils import save_numpy_array , save_object
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline


class DataTransformation():
    def __init__(self , data_validation_artifact:DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def read_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformation_object(self) -> KNNImputer:
        try:
            imputer = KNNImputer(**self.data_transformation_config.impute_params)
            pipeline = Pipeline([("imputer", imputer)])
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def intiate_date_transformation(self) -> None:
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            
            input_featire_train_df = train_df.drop(columns=[self.data_transformation_config.target_column] , axis =1)
            target_feature_train_df = train_df[self.data_transformation_config.target_column].replace(-1 , 0)

            input_featire_test_df = test_df.drop(columns=[self.data_transformation_config.target_column] , axis =1)
            target_feature_test_df = test_df[self.data_transformation_config.target_column].replace(-1 , 0)
            
            data_transformation_object = self.get_data_transformation_object()
            
            transformed_train_array = data_transformation_object.fit_transform(input_featire_train_df)
            transformed_test_array = data_transformation_object.transform(input_featire_test_df)

            train_array = np.c_[transformed_train_array , np.array(target_feature_train_df)]
            test_array = np.c_[transformed_test_array , np.array(target_feature_test_df)]
            
            save_numpy_array(file_path=self.data_transformation_config.transformed_train_file_path, array=train_array)
            save_numpy_array(file_path=self.data_transformation_config.transformed_test_file_path, array=test_array)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path, obj=data_transformation_object)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path)

            
            logging.info("Data transformation completed successfully.")

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)