# External imports
import lightning as L
import pandas as pd
import os
from torch.utils.data import DataLoader
import tempfile
import numpy as np

# Internal imports
from .dataset_rdkit_features import DEEPScreenDatasetPredict, DEEPScreenDatasetTrain, DEEPScreenDatasetTest
from utils.logging_deepscreen import logger
from utils.exceptions import InvalidDataframeException
from utils.configurations import configs


class DEEPscreenDataModule(L.LightningDataModule):
    
    def __init__(self, data:pd.DataFrame, features:np.ndarray, batch_size:int, experiment_result_path:str, data_split_mode:str, tmp_imgs:bool = False):

        super(DEEPscreenDataModule, self).__init__()

        self.save_hyperparameters()
    
        self.result_path = experiment_result_path

        self.batch_size = batch_size

        self.features = features

        if tmp_imgs:
            self.imgs_path = tempfile.TemporaryDirectory().name
        else:
            self.imgs_path = os.path.join(self.result_path,"imgs")
        
        if data_split_mode in ("random_split","non_random_split","scaffold_split","predict"):
            self.data_split = data_split_mode
        else:
            raise Exception("data split mode should be one of random_split/non_random_split/scaffold_split/predict")

        self.data = data

        if data_split_mode != "predict" and not {"comp_id","smiles","bioactivity"}.issubset(set(self.data.columns)):
            logger.error("invalid columns of df")
            raise InvalidDataframeException("must contain the following columns {'comp_id','smiles','bioactivity'}")

    def setup(self,stage:str):
        
        # cleaning data to prune posible errors
        self.data = self.data.dropna(how="any")

        logger.info(f"Using a total of {len(self.data)} datapoints")

        if stage == "fit" or stage == "test" or stage == "validation":

            # sanitizeing data
            self.data["bioactivity"] = self.data["bioactivity"].astype(int)

            if self.data_split == "random_split":
                #TODO
                raise NotImplementedError

            if self.data_split == "non_random_split":
                dataset = self._non_random_split(self.data,self.features)
                self.train, self.train_features = dataset["train"]
                self.validate, self.validate_features = dataset["validation"]
                self.test, self.test_features  = dataset["test"]

            if self.data_split == "scaffold_split":
                #TODO
                raise NotImplementedError
            
        
        if stage == "predict":
            self.predict = self.data
            self.predict_features = self.features
    
    def get_number_training_batches(self):
        if self.data_split == "non_random_split":
            number_training_batches = round(len(self.data[self.data["data_split"] == "train"])/self.batch_size)
        else:
            number_training_batches = 50
            
        return number_training_batches
    
    def _non_random_split(self,data,features):
        if "data_split" not in data.columns:
            logger.error("theres not a 'data_split' column in the dataframe for non random datasplit")
            raise InvalidDataframeException("Missing 'data_split' column while using non_random_split")

        if not set(data["data_split"].unique()).issubset({"train","validation","test","predict"}):
            logger.error("invalid tags or missing tags for data spliting in non random datasplit")
            raise InvalidDataframeException("Invalid tags or missing tags for data spliting in non random datasplit, tags should be in ('trian','validation','test','predict')")
        
        dataset = {"train":None,"validation":None,"test":None}
        
        try:
            for key in data["data_split"].unique():
                dataset[key] = (data[data["data_split"]==key], features[data["data_split"]==key]) # this im gonna did with features is horrible, but im tired and i need to try features TODO improve

                logger.info(f"non_random_split dataset splited {key}={len(dataset[key][0])}")
            
        except Exception as e:
            logger.error(f"Unable to create non_random_split datasets {e}")
            raise RuntimeError("non_random_split dataloader type generator failed")

        return dataset


    def train_dataloader(self):
        self.training_dataset = DEEPScreenDatasetTrain(self.imgs_path, self.train, self.train_features)
        return DataLoader(self.training_dataset,batch_size=self.hparams.batch_size,shuffle=True)
    
    def val_dataloader(self):
        self.validation_dataset = DEEPScreenDatasetTest(self.imgs_path, self.validate, self.validate_features)
        return DataLoader(self.validation_dataset,batch_size=self.hparams.batch_size)
    
    def test_dataloader(self):
        self.test_dataset = DEEPScreenDatasetTest(self.imgs_path, self.test, self.test_features)
        return DataLoader(self.test_dataset,batch_size=self.hparams.batch_size)
    
    def predict_dataloader(self):
        self.predict_dataset = DEEPScreenDatasetPredict(self.imgs_path, self.predict, self.predict_features)
        return DataLoader(self.predict_dataset,batch_size=self.hparams.batch_size)

        