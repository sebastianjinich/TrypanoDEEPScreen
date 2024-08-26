from utils.logging_deepscreen import logger

class InvalidDataframeException(Exception):
    def __init__(self,message):
        self.message = message
        logger.critical(f"Dataframe not valid {self.message}")