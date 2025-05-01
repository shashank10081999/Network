import os
import sys
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
import json
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")

from pymongo import MongoClient

class NetworkDataExtract():
    def __init(delf):
        try:
            pass
        except Exception as e:
                raise NetworkSecurityException(e , sys)
    
    def csv_to_json(self , file_path):
        try:
              data = pd.read_csv(file_path)
              data.reset_index(drop=True , inplace=True)
              records = (data.to_json(orient="records"))
              return list(json.loads(records))
        except Exception as e:
            raise NetworkSecurityException(e , sys)
         
    def insert_data_into_mongo_db(self , data , database , collection):
        try:
            client = MongoClient(mongo_db_url)
            db = client[database]
            collection = db[collection]
            collection.insert_many(data)
            logging.info("Data inserted into MongoDB successfully.")
        except Exception as e:
            raise NetworkSecurityException(e , sys)


if __name__ == "__main__":
     data_object = NetworkDataExtract()
     records = data_object.csv_to_json("network_data\phisingData.csv")
     print(records[0])
     data_object.insert_data_into_mongo_db(records , "NetworkData_livesmell" , "Network")