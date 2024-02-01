import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import os
from torch.utils.data import DataLoader
import tempfile
from dataset import DEEPScreenDatasetPredict, DEEPScreenDatasetTrain
from ..utils.logging_deepscreen import logger
from ..utils.exceptions import InvalidDataframeException
from ..utils.configurations import configs


class DEEPscreenDataModule(L.LightningDataModule):
    
    def __init__(self, data:pd.DataFrame, target_id:str, experiment_result_path:str, data_split_mode:str, tmp_imgs:bool = False):

        super.__init__()
    
        self.result_path = experiment_result_path

        if tmp_imgs:
            self.imgs_path = tempfile.TemporaryDirectory().name
        else:
            self.imgs_path = os.path.join(self.result_path,target_id,"imgs")
        
        if data_split_mode in ("random_split","non_random_split","scaffold_split","predict"):
            self.data_split = data_split_mode
        else:
            raise Exception("data split mode sholud be one of random_split/non_random_split/scaffold_split")

        self.data = data

        if not {"comp_id","smiles","bioactivity"}.issubset(set(self.data.columns)):
            logger.error("invalid columns of df")
            raise InvalidDataframeException


    def setup(self):
        
        # cleaning data to prune posible errors
        self.data = self.data.dropna(how="any")

        logger.info(f"Using a total of {len(self.data)} datapoints")

        if self.data_split == "random_split":
            #TODO
            raise NotImplementedError    
        
        if self.data_split == "non_random_split":
            if "data_split" not in self.data.columns:
                logger.error("theres not a 'data_split' column in the dataframe for non random datasplit")
                raise InvalidDataframeException("Missing 'data_split' column while using non_random_split")
            
            if set(self.data["data_split"].unique()) != {"train","test","validation"}:
                logger.error("invalid tags or missing tags for data spliting in non random datasplit")
                raise InvalidDataframeException("Invalid tags or missing tags for data spliting in non random datasplit")
            
            try:
                self.train = self.data[self.data["data_split"] == "train"]
                self.validate = self.data[self.data["data_split"] == "validation"]
                self.test = self.data[self.data["data_split"] == "test"]
                logger.info(f"non_random_split datasets splited train={len(self.train)},validation={len(validate)},test={len(test)}")
            except Exception as e:
                logger.error(f"Unable to create non_random_split datasets {e}")
                raise RuntimeError("non_random_split dataloader type generator failed")
            
        if self.data_split == "scaffold_split":
            #TODO
            raise NotImplementedError
        

    def train_dataloader(self):
        hyperparameters = configs.get_hyperparameters()
        self.training_dataset = DEEPScreenDatasetTrain(self.imgs_path, self.train)
        
        return DataLoader(self.training_dataset,batch_size=hyperparameters["batch_size"])
    
    def val_dataloader(self):
        hyperparameters = configs.get_hyperparameters()
        self.validation_dataset = DEEPScreenDatasetTrain(self.imgs_path, self.validate)
        return DataLoader(self.validation_dataset,batch_size=hyperparameters["batch_size"])
    
    def test_dataloader(self):
        hyperparameters = configs.get_hyperparameters()
        self.predict_dataset = DEEPScreenDatasetTrain(self.imgs_path, self.test)
        return DataLoader(self.test_dataset,batch_size=hyperparameters["batch_size"])
    
    def predict_dataloader(self):
        hyperparameters = configs.get_hyperparameters()
        self.predict_dataset = DEEPScreenDatasetPredict(self.imgs_path, self.data)
        return DataLoader(self.predict_dataset,batch_size=hyperparameters["batch_size"])
        
