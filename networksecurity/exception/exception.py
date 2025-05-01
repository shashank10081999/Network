import sys 
import os
from networksecurity.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self , error_message , error_details:sys):
        self.error_message = error_message
        _ , _ , exe_info = error_details.exc_info()
        self.lineno = exe_info.tb_lineno
        self.filename = exe_info.tb_frame.f_code.co_filename
    
    def __str__(self):
        return f"Error occured in python script name {self.filename} in line {self.lineno}"

if __name__ == "__main__":
    try:
        logger.logging.info("logging testing")
        a = 1/0
    except Exception as e:
        raise NetworkSecurityException(e , sys)