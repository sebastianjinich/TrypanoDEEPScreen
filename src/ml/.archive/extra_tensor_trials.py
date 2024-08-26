# %%
# External imports
import torch
from torch.nn import functional as F
import torch.nn as nn
import lightning as L
from torchmetrics import classification, MetricCollection
import pandas as pd
import numpy as np
import os

# Internal imports
from utils.configurations import configs
from utils.logging_deepscreen import logger

class DEEPScreenClassifier(L.LightningModule):
    def __init__(self,fully_layer_1, fully_layer_2, drop_rate, learning_rate, batch_size, experiment_result_path, target):
        super(DEEPScreenClassifier, self).__init__()
        self.save_hyperparameters()
        logger.info(f"Using hyperparameters {[i for i in self.hparams.items()]}") 

        # Model architecture
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 2)
        self.bn5 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1010, fully_layer_1)
        self.fc2 = nn.Linear(fully_layer_1, fully_layer_2)
        self.fc3 = nn.Linear(fully_layer_2, 2)
        self.drop_rate = drop_rate

        # Object atributes
        self.config = configs
        
        # Performance trackers
        self.train_metrics = MetricCollection(
             {
                "train_acc": classification.BinaryAccuracy(threshold=0.5),
                "train_prec": classification.BinaryPrecision(threshold=0.5),
                "train_f1": classification.BinaryF1Score(threshold=0.5),
                "train_mcc": classification.BinaryMatthewsCorrCoef(threshold=0.5),
                "train_recall": classification.BinaryRecall(threshold=0.5),
                "train_auroc": classification.BinaryAUROC(),
                "train_auroc_15": classification.BinaryAUROC(max_fpr=0.15),
                "train_calibration_error": classification.BinaryCalibrationError()
             }
         )
        
        self.val_metrics = MetricCollection(
             {
                "val_acc": classification.BinaryAccuracy(threshold=0.5),
                "val_prec": classification.BinaryPrecision(threshold=0.5),
                "val_f1": classification.BinaryF1Score(threshold=0.5),
                "val_mcc": classification.BinaryMatthewsCorrCoef(threshold=0.5),
                "val_recall": classification.BinaryRecall(threshold=0.5),
                "val_auroc": classification.BinaryAUROC(),
                "val_auroc_15": classification.BinaryAUROC(max_fpr=0.15),
                "val_calibration_error": classification.BinaryCalibrationError()
             }
         )
        
        self.test_metrics = MetricCollection(
             {
                "test_acc": classification.BinaryAccuracy(threshold=0.5),
                "test_prec": classification.BinaryPrecision(threshold=0.5),
                "test_f1": classification.BinaryF1Score(threshold=0.5),
                "test_mcc": classification.BinaryMatthewsCorrCoef(threshold=0.5),
                "test_recall": classification.BinaryRecall(threshold=0.5),
                "test_auroc": classification.BinaryAUROC(),
                "test_auroc_15": classification.BinaryAUROC(max_fpr=0.15),
                "test_calibration_error": classification.BinaryCalibrationError()
             }
         )

        # Predictions
        self.test_predictions = pd.DataFrame()
        self.predictions = pd.DataFrame()

    def forward(self, x, descriptors_features):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 32*5*5)
        # Using MinMax Scaleing to scale down the descriptors features to match the MinMax of the output of the convolutions
        min = x.min()
        max = x.max()
        descriptors_features_scaled = descriptors_features * (max - min) + min
        x = torch.cat((x, descriptors_features_scaled), dim=1) 
        x = F.dropout(F.relu(self.fc1(x)), self.drop_rate, training = self.training)
        x = F.dropout(F.relu(self.fc2(x)), self.drop_rate, training = self.training)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.hparams.learning_rate)
        return optimizer
    
    def cross_entropy_loss(self,input,labels):
        return F.cross_entropy(input,labels)
    
    def training_step(self,train_batch,batch_idx):
        img_arrs, label, comp_id, descriptor_features = train_batch
        y_pred = self.forward(img_arrs,descriptor_features)
        y_pred_soft_max = F.softmax(y_pred,dim=1)
        y_pred_soft_max_1_active = y_pred_soft_max[:,1]
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        self.log('train_loss', loss, batch_size=self.hparams.batch_size, prog_bar=True, sync_dist=True)

        output = self.train_metrics(y_pred_soft_max_1_active,label.int())
        self.log_dict(output,on_step=False,on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
        
        return loss
      
    def validation_step(self,val_batch,batch_idx):
        img_arrs, label, comp_id, descriptor_features  = val_batch
        y_pred = self.forward(img_arrs, descriptor_features)
        y_pred_soft_max = F.softmax(y_pred,dim=1)
        y_pred_soft_max_1_active = y_pred_soft_max[:,1]
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        self.log('val_loss', loss, batch_size=self.hparams.batch_size, prog_bar=True, sync_dist=True)

        output = self.val_metrics(y_pred_soft_max_1_active,label.int())
        self.log_dict(output,on_step=False,on_epoch=True,batch_size=self.hparams.batch_size,  sync_dist=True)

    def test_step(self,test_batch,batch_idx):
        img_arrs, label, comp_id, descriptor_features = test_batch
        y_pred = self.forward(img_arrs, descriptor_features)
        _, preds = torch.max(y_pred,1)
        y_pred_soft_max = F.softmax(y_pred,dim=1)
        y_pred_soft_max_1_active = y_pred_soft_max[:,1]
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size, prog_bar=True)

        output = self.test_metrics(y_pred_soft_max_1_active,label.int())
        self.log_dict(output,on_step=False,on_epoch=True,batch_size=self.hparams.batch_size)

        comp_id_pd = pd.Series(comp_id,name="comp_id")
        label_pd = pd.Series(label.cpu(),name="label")
        pred_pd = pd.Series(preds.cpu(),name="prediction")
        pred_0_pd = pd.Series(y_pred_soft_max[:,0].cpu(),name="0_inactive_probability")
        pred_1_pd = pd.Series(y_pred_soft_max[:,1].cpu(),name="1_active_probability")
        batch_predictions = pd.concat([comp_id_pd,label_pd,pred_pd,pred_0_pd,pred_1_pd],axis=1)
        self.test_predictions = pd.concat([self.test_predictions,batch_predictions],axis=0)

    def on_test_end(self):
        self.test_predictions.to_csv(os.path.join(self.hparams.experiment_result_path,f"test_{self.hparams.target}_{self.hparams.fully_layer_1}-{self.hparams.fully_layer_2}-{self.hparams.learning_rate}-{self.hparams.drop_rate}-{self.hparams.batch_size}.csv"),index=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img_arrs, comp_id, descriptor_features = batch
        y_pred = self.forward(img_arrs, descriptor_features)
        _, preds = torch.max(y_pred,1)
        comp_id_pd = pd.Series(comp_id,name="comp_id")
        pred_pd = pd.Series(preds.cpu(),name="prediction")
        y_pred_soft_max = F.softmax(y_pred,dim=1) 
        pred_0_pd = pd.Series(y_pred_soft_max[:,0].cpu(),name="0_inactive_probability")
        pred_1_pd = pd.Series(y_pred_soft_max[:,1].cpu(),name="1_active_probability")
        batch_predictions = pd.concat([comp_id_pd,pred_pd,pred_0_pd,pred_1_pd],axis=1)
        self.predictions = pd.concat([self.predictions,batch_predictions],axis=0)
        return batch_predictions

    def on_predict_epoch_end(self):
        return self.predictions
    
    def on_predict_end(self):
        self.predictions.to_csv(os.path.join(self.hparams.experiment_result_path,f"predictions_{self.hparams.target}_{self.hparams.fully_layer_1}-{self.hparams.fully_layer_2}-{self.hparams.learning_rate}-{self.hparams.drop_rate}-{self.hparams.batch_size}.csv"),index=False)

# %%
# External imports
import os
import pandas as pd
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
import torch
from rdkit.Chem import Draw, MolFromSmiles
import re
import deepchem as dc

# Internal Imports
from utils.constants import RANDOM_STATE
from utils.logging_deepscreen import logger
from utils.configurations import configs

random.seed(RANDOM_STATE)

class DEEPScreenDataset(Dataset):
    def __init__(self, path_imgs_files:str, df_compid_smiles_bioactivity:pd.DataFrame):
            super(DEEPScreenDataset, self).__init__()

            self.path_imgs = path_imgs_files
            self.df = df_compid_smiles_bioactivity.copy(deep=True)
            self.config = configs

            if not os.path.exists(self.path_imgs):
                os.makedirs(self.path_imgs)

            # creating molecules images -> path will be stored in 'img_molecule' column
            self.df['img_molecule'] = self.df.apply(lambda x: self.smiles_to_img_png(x["comp_id"],x["smiles"],self.path_imgs),axis=1)

            featurizer = dc.feat.RDKitDescriptors()
            
            rdkit_features_not_std = featurizer.featurize(self.df["smiles"])
            
            # min max standarization for min max scaleing to append featurese to cnn output
            rdkit_features_not_std_max = rdkit_features_not_std.max(axis=1,keepdims=True)
            rdkit_features_not_std_min = rdkit_features_not_std.min(axis=1,keepdims=True)
            self.rdkit_features = (rdkit_features_not_std - rdkit_features_not_std_min) / (rdkit_features_not_std_max - rdkit_features_not_std_min)


            logger.debug(f'Dataset created in {self.path_imgs}')


    def __len__(self):
        return len(self.df.index)
    
    def smiles_to_img_png(self, comp_id:str, smiles:str, output_path:str)->str:
        '''
        Given an id and an output path the function will create a image with the 2D molecule. 

        Returns: the path to the file i.e. /output_path/comp_id.png
        '''
        mol = MolFromSmiles(smiles)
        opt = self.config.get_mol_draw_options()
        img_size = self.config.get_img_size()
        comp_id_clean = re.sub('[^A-Za-z0-9]+', '_', comp_id)

        output_file = os.path.join(output_path, f"{comp_id_clean}.png")
        Draw.MolToFile(mol, output_file, size=img_size, options=opt)
        return output_file


class DEEPScreenDatasetPredict(DEEPScreenDataset):
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row["comp_id"]
        img_path = row['img_molecule']
        features = self.rdkit_features[index]
        img_arr = cv2.imread(img_path)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        return torch.tensor(img_arr).type(torch.FloatTensor), comp_id, torch.tensor(features).type(torch.FloatTensor)
    

class DEEPScreenDatasetTrain(DEEPScreenDataset):

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row["comp_id"]
        img_path = row['img_molecule']
        features = self.rdkit_features[index]
        img_arr = cv2.imread(img_path)
        if random.random()>=0.50:
            angle = random.randint(0,359)
            rows, cols, channel = img_arr.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_arr = cv2.warpAffine(img_arr, rotation_matrix, (cols, rows), cv2.INTER_LINEAR,
                                             borderValue=(255, 255, 255))  # cv2.BORDER_CONSTANT, 255)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        label = row["bioactivity"]

        return torch.tensor(img_arr).type(torch.FloatTensor), torch.tensor(label).type(torch.LongTensor), comp_id, torch.tensor(features).type(torch.FloatTensor)
    

class DEEPScreenDatasetTest(DEEPScreenDataset):

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row["comp_id"]
        img_path = row['img_molecule']
        features =  self.rdkit_features[index]
        img_arr = cv2.imread(img_path)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        label = row["bioactivity"]

        return torch.tensor(img_arr).type(torch.FloatTensor), torch.tensor(label).type(torch.LongTensor), comp_id, torch.tensor(features).type(torch.FloatTensor)

# %%



# %%
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


# %%
df = pd.read_csv("/home/sjinich/disco/TrypanoDEEPscreen/.data/processed/CHEMBL2581.csv")

# %%
trainer = Trainer(max_epochs=100)
model = DEEPScreenClassifier(fully_layer_1=256,fully_layer_2=32,drop_rate=0.5,learning_rate=0.0001,batch_size=32,experiment_result_path="../../.experiments/chembl2581",target="CHEMBL2581")
datamodule = DEEPscreenDataModule(data=df,batch_size=32,experiment_result_path="../../.experiments/chembl2581",data_split_mode="non_random_split",tmp_imgs=True)
trainer.fit(model,datamodule=datamodule)

# %%
img_arr = cv2.imread("/home/sjinich/disco/TrypanoDEEPscreen/.experiments/imgs/2_Deoxy_2_methyl_snitroso_carbamoyl_amino_hexose.png")
if random.random()>=0:
    angle = random.randint(0,359)
    rows, cols, channel = img_arr.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_arr = cv2.warpAffine(img_arr, rotation_matrix, (cols, rows), cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255))  # cv2.BORDER_CONSTANT, 255)
img_arr = np.array(img_arr) / 255.0
img_arr = img_arr.transpose((2, 0, 1))
tensor = torch.tensor([img_arr]).type(torch.FloatTensor)

network = DEEPScreenClassifier(512,256,0.001,0.3,32,"/home/sjinich/disco/TrypanoDEEPscreen/.experiments/trial_adding_vector","trial_adding_tensor")

otuput, lineal_tensor = network.forward(tensor, df_d.loc[0,"descriptor_features"])

# %%
otuput.squeeze()

# %%
featurizer = dc.feat.RDKitDescriptors()
features = featurizer.featurize(df["smiles"])
descriptors_rdkit = pd.Series([torch.tensor(tensor).type(torch.FloatTensor) for tensor in features], name="descriptor_features")
df = pd.concat([df,descriptors_rdkit],axis=1)


# %%
df.iloc[0]["descriptor_features"]


