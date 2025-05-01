import logging
import logging.config
import os
from datetime import datetime

log_file = f"{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.log"
log_dir  = os.path.join(os.getcwd(), "logs")
log_path = os.path.join(log_dir , log_file)
os.makedirs(log_dir , exist_ok=True)
logging.basicConfig(
    filename = log_path,
    format="[%(asctime)s] %(lineno)d %(name)s _ %(levelname)s-%(message)s", 
    level=logging.INFO 
)