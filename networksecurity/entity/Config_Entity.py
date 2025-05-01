from datetime import datetime
from networksecurity.contants import training_pipeline
import os
import sys

class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_dir_name = training_pipeline.ARTIFACT_DIR
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.artifact_dir = os.path.join(self.artifact_dir_name, self.timestamp)
        self.train_file_name = training_pipeline.TRAIN_FILE_NAME
        self.test_file_name = training_pipeline.TEST_FILE_NAME
        self.target_column = training_pipeline.TARGET_COLUMN
        self.data_ingestion_collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.data_ingestion_database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.data_ingestion_dir_name = training_pipeline.DATA_INGESTION_DIR_NAME
        self.data_ingestion_feature_store_dir = training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR
        self.data_ingestion_ingested_dir = training_pipeline.DATA_INGESTION_INGESTED_DIR
        self.data_ingestion_train_test_split_ration = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.data_ingestion_file_name = training_pipeline.FILE_NAME

        self.data_validation_dir_name = training_pipeline.DATA_VALIDATION_DIR_NAME
        self.data_validation_valid_dir = training_pipeline.DATA_VALIDATION_VALID_DIR
        self.data_validation_invalid_dir = training_pipeline.DATA_VALIDATION_INVALID_DIR
        self.data_validation_drift_report_dir = training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR
        self.data_validation_drift_report_file_name = training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        self.data_validation_report_file_name = training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME

        self.data_transformation_dir_name = training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        self.data_transformation_transformed_dir = training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR
        self.data_transformation_transformed_object_dir = training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
        self.data_transformation_transformed_object_file_name = training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME
        self.data_transformation_impute_params = training_pipeline.DATA_TRANSFORMATION_IMPUTE_PARAMS


        self.model_trainer_dir_name = training_pipeline.MODEL_TRAINER_DIR_NAME
        self.model_trainer_trained_model_dir = training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR
        self.model_trainer_trained_model_file_name = training_pipeline.MODEL_TRAINER_TRAINED_MODEL_FILE_NAME
        self.model_trainer_expected_accuracy = training_pipeline.MODEL_TRAINER_EXPECTED_ACCURACY
        self.model_trainer_over_fitting_under_fitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        self.model_trainer_mlflow_experiment_name = training_pipeline.MODEL_TRAINER_MLFLOW_EXPERIMENT_NAME

    
class DataIngestionConfig():
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline_config.data_ingestion_dir_name)
        self.feature_store_dir = os.path.join(self.data_ingestion_dir, training_pipeline_config.data_ingestion_feature_store_dir)
        self.ingested_dir = os.path.join(self.data_ingestion_dir, training_pipeline_config.data_ingestion_ingested_dir)
        self.train_file_path = os.path.join(self.ingested_dir, training_pipeline_config.train_file_name)
        self.test_file_path = os.path.join(self.ingested_dir, training_pipeline_config.test_file_name)
        self.data_ingestion_database_name = training_pipeline_config.data_ingestion_database_name
        self.data_ingestion_collection_name = training_pipeline_config.data_ingestion_collection_name
        self.feature_store_file_path = os.path.join(self.feature_store_dir, training_pipeline_config.data_ingestion_file_name)
        self.data_ingestion_train_test_split_ration = training_pipeline_config.data_ingestion_train_test_split_ration

class DataValidationConfig():
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline_config.data_validation_dir_name)
        self.validated_dir = os.path.join(self.data_validation_dir, training_pipeline_config.data_validation_valid_dir)
        self.invalid_dir = os.path.join(self.data_validation_dir, training_pipeline_config.data_validation_invalid_dir)
        self.valid_train_file_path = os.path.join(self.validated_dir, training_pipeline_config.train_file_name)
        self.invalid_train_file_path = os.path.join(self.invalid_dir, training_pipeline_config.train_file_name)
        self.valid_test_file_path = os.path.join(self.validated_dir, training_pipeline_config.test_file_name)
        self.invalid_test_file_path = os.path.join(self.invalid_dir, training_pipeline_config.test_file_name)
        self.drift_report_dir = os.path.join(self.data_validation_dir, training_pipeline_config.data_validation_drift_report_dir)
        self.drift_report_file_path = os.path.join(self.drift_report_dir, training_pipeline_config.data_validation_drift_report_file_name)

class DataTransformationConfig():
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.target_column = training_pipeline_config.target_column
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline_config.data_transformation_dir_name)
        self.transformed_dir = os.path.join(self.data_transformation_dir, training_pipeline_config.data_transformation_transformed_dir)
        self.transformed_object_dir = os.path.join(self.data_transformation_dir, training_pipeline_config.data_transformation_transformed_object_dir)
        self.transformed_object_file_path = os.path.join(self.transformed_object_dir, training_pipeline_config.data_transformation_transformed_object_file_name)
        self.transformed_train_file_path = os.path.join(self.transformed_dir, training_pipeline_config.train_file_name.replace(".csv" , ".npy"))
        self.transformed_test_file_path = os.path.join(self.transformed_dir, training_pipeline_config.test_file_name.replace(".csv" , ".npy"))
        self.impute_params = training_pipeline_config.data_transformation_impute_params

class ModelTrainerConfig():
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline_config.model_trainer_dir_name)
        self.trained_model_dir = os.path.join(self.model_trainer_dir, training_pipeline_config.model_trainer_trained_model_dir)
        self.trained_model_file_path = os.path.join(self.trained_model_dir, training_pipeline_config.model_trainer_trained_model_file_name)
        self.expected_accuracy = training_pipeline_config.model_trainer_expected_accuracy
        self.over_fitting_under_fitting_threshold = training_pipeline_config.model_trainer_over_fitting_under_fitting_threshold
        self.mlflow_experiment_name = training_pipeline_config.model_trainer_mlflow_experiment_name