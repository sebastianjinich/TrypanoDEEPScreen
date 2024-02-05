import logging
import datetime
import os

path = "./log"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)

logger = logging.getLogger("lightning.pytorch")
#logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)-8s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

date_now = str(datetime.datetime.now())[:19].replace(' ','_')

file_handler_debug = logging.FileHandler(os.path.join(path,f'{date_now}_debug.log'))
file_handler_debug.setLevel(logging.DEBUG)
file_handler_debug.setFormatter(formatter)

file_handler_info = logging.FileHandler(os.path.join(path,f'{date_now}_info.log'))
file_handler_info.setLevel(logging.INFO)
file_handler_info.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
    
logger.addHandler(file_handler_debug)
logger.addHandler(file_handler_info)
logger.addHandler(console_handler)