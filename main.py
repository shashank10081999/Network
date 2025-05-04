import os
import sys
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
import json
from dotenv import load_dotenv
import pandas as pd
from pymongo import MongoClient
load_dotenv()
import fastapi
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from fastapi.responses import RedirectResponse 
mongo_db_url = os.getenv("MONGO_DB_URL")
from networksecurity.contants import training_pipeline
from networksecurity.cloud.aws import syn_folder_s3
from networksecurity.utils.utils import load_object , save_object , load_numpy_array
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.compoments.data_ingestion import DataIngestion
from networksecurity.compoments.data_validation import DataValidation
from networksecurity.compoments.model_trainer import ModelTrainer
from networksecurity.compoments.data_transformation import DataTransformation
from networksecurity.entity.Config_Entity import TrainingPipelineConfig , DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact , ModelTrainerArtifact

def main():
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

        logging.info("Model training completed successfully.")

        syn_folder_s3(bucket_name=training_pipeline.AWS_BUCKET_NAME, folder_path=os.path.dirname(model_trainer_artifact.best_model_file_path))

        syn_folder_s3(bucket_name=training_pipeline.AWS_BUCKET_NAME, folder_path=training_pipeline_config.artifact_dir)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

app  = FastAPI()
origin = ["*"]

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train():
    try:
        main()
        return {"message": "Training completed successfully."}
    except Exception as e:
        return NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict(request: fastapi.Request , uploadFile: fastapi.UploadFile = fastapi.File(...)):
    try:
        df = pd.read_csv(uploadFile.file)
        preproccessor = load_object(object_file_path=os.path.join("final_model", "preprocessor.pkl"))
        model = load_object(object_file_path=os.path.join("final_model", "model.pkl"))
        model_instance = NetworkModel(preprocessor=preproccessor, model=model)
        predictions = model_instance.predict(df.drop(columns=["Result"]))
        print(predictions)
        predictions = pd.DataFrame(predictions, columns=["Prediction"])
        os.makedirs("prediction_output", exist_ok=True)
        # Save the predictions to a CSV file
        predictions.to_csv("prediction_output/predictions.csv", index=False)

        return {"message": "Prediction completed successfully."}
    except Exception as e:
        return NetworkSecurityException(e, sys)


if __name__ == "__main__":
   app_run(app,host="0.0.0.0", port=8080)