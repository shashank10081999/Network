from networksecurity.contants import training_pipeline
from networksecurity.entity.Config_Entity import TrainingPipelineConfig , DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact , ModelTrainerArtifact
import os
import sys
import mlflow
import mlflow.sklearn
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
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.utils import load_object , save_object , load_numpy_array
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier)
from networksecurity.utils.ml_utils.model.estimator import evaluate_model
import dagshub
dagshub.init(repo_owner=os.getenv("REPO_OWNER"), repo_name=os.getenv("REPO_NAME"), mlflow=True)


class ModelTrainer():
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, model, classification_train_metric, classification_test_metric):
        try:
            mlflow.set_experiment(self.model_trainer_config.mlflow_experiment_name)

            with mlflow.start_run():
                mlflow.log_param("model_type", type(model).__name__)
                mlflow.log_param("train_accuracy", classification_train_metric.precision)
                mlflow.log_param("test_accuracy", classification_test_metric.precision)
                mlflow.log_param("train_f1_score", classification_train_metric.f1_score)
                mlflow.log_param("test_f1_score", classification_test_metric.f1_score)
                mlflow.log_param("train_recall", classification_train_metric.recall)
                mlflow.log_param("test_recall", classification_test_metric.recall)

                mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def model_train(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "LogisticRegression": LogisticRegression(verbose=1),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(verbose=1),
                "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
                "AdaBoostClassifier": AdaBoostClassifier()
            }

            parameters = {
                "LogisticRegression": {
                    'C': [0.1, 1, 10],
                    'max_iter': [100, 200, 300]
                },
                "DecisionTreeClassifier": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30]
                },
                "RandomForestClassifier": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30]
                },
                "GradientBoostingClassifier": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                }
            }

            model_report = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model=models, params=parameters)

            best_model_name = sorted(model_report.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)[0][0]
            best_model = models[best_model_name]

            classification_train_metric = get_classification_metric(y_train, best_model.predict(x_train))
            classification_test_metric = get_classification_metric(y_test, best_model.predict(x_test))

            self.track_mlflow(best_model , classification_train_metric , classification_test_metric)

            
            logging.info("Tracking MLflow metrics for the best model.")

            preprocessor = load_object(object_file_path=self.data_transformation_artifact.transformed_object_file_path)
            model = NetworkModel(preprocessor=preprocessor, model=best_model)

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=model)

            os.makedirs("final_model", exist_ok=True)
            best_model_file_path = os.path.join("final_model", "model.pkl")
            save_object(file_path=best_model_file_path, obj=best_model)
            save_object(file_path=os.path.join("final_model", "preprocessor.pkl"), obj=preprocessor)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                trained_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
                best_model_file_path=best_model_file_path
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise NetworkSecurityException(e, sys)


    
    def initiate_model_trainer(self):
        try:
            # Load the transformed data
            train_arr = load_numpy_array(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(file_path=self.data_transformation_artifact.transformed_test_file_path)

            x_train , y_train , x_test , y_test = (

                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model_trainer_artifact = self.model_train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            return model_trainer_artifact


        except Exception as e:
            raise NetworkSecurityException(e, sys)