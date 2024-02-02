from utils.logging_deepscreen import logger

class InvalidDataframeException(Exception):
    def __init__(self,message):
        super(InvalidDataframeException, self).__init__()
        logger.critical(f"Dataframe not valid {self.message}")