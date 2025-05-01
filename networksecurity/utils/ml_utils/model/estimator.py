from networksecurity.contants.training_pipeline import SAVED_MODEL_DIR , MODEL_TRAINER_TRAINED_MODEL_FILE_NAME

import os 
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score , f1_score


def evaluate_model(x_train, y_train, x_test, y_test , model, params):
    try:
        report = {}
        for i in range(len(model)):
            model_name = list(model.keys())[i]
            model_instance = model[model_name]
            param = params[model_name]
            
            
            grid_search = GridSearchCV(estimator=model_instance, param_grid=param, cv=5)
            grid_search.fit(x_train, y_train)
            
            model_instance.set_params(**grid_search.best_params_)
            model_instance.fit(x_train, y_train)

            y_train_pred = model_instance.predict(x_train)
            y_test_pred = model_instance.predict(x_test)

            
            report[model_name] = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "best_params": grid_search.best_params_
            }

        return report
    except Exception as e:
        raise NetworkSecurityException(e , sys)

class NetworkModel():
    def __init__(self , preprocessor , model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e , sys)
        
    def predict(self , X):
        try:
            preprocessed_data = self.preprocessor.transform(X)
            predictions = self.model.predict(preprocessed_data)
            return predictions
        except Exception as e:
            raise NetworkSecurityException(e , sys)
        
