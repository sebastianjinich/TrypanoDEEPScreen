from logging_deepscreen import logger

class InvalidDataframeException(Exception):
    def __init__(self,message):
        super().__init__(message)
        logger.critical(f"Dataframe not valid {self.message}")