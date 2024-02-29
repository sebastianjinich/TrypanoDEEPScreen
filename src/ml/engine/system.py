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
        self.fc1 = nn.Linear(32*5*5, fully_layer_1)
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

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 32*5*5)
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
        img_arrs, label, comp_id = train_batch
        y_pred = self.forward(img_arrs)
        y_pred_soft_max = F.softmax(y_pred,dim=1)
        y_pred_soft_max_1_active = y_pred_soft_max[:,1]
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        self.log('train_loss', loss, batch_size=self.hparams.batch_size, prog_bar=True, sync_dist=True)

        output = self.train_metrics(y_pred_soft_max_1_active,label.int())
        self.log_dict(output,on_step=False,on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
        self.train_metrics.update(y_pred_soft_max_1_active,label.int())
        
        return loss
    
    def on_training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True, batch_size=self.hparams.batch_size,  sync_dist=True)
        self.train_metrics.reset()
    
    def validation_step(self,val_batch,batch_idx):
        img_arrs, label, comp_id = val_batch
        y_pred = self.forward(img_arrs)
        y_pred_soft_max = F.softmax(y_pred,dim=1)
        y_pred_soft_max_1_active = y_pred_soft_max[:,1]
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        self.log('val_loss', loss, batch_size=self.hparams.batch_size, prog_bar=True, sync_dist=True)

        output = self.val_metrics(y_pred_soft_max_1_active,label.int())
        self.log_dict(output,on_step=False,on_epoch=True,batch_size=self.hparams.batch_size,  sync_dist=True)
        self.val_metrics.update(y_pred_soft_max_1_active,label.int())
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True,batch_size=self.hparams.batch_size,  sync_dist=True)
        self.val_metrics.reset()

    def test_step(self,test_batch,batch_idx):
        img_arrs, label, comp_id = test_batch
        y_pred = self.forward(img_arrs)
        _, preds = torch.max(y_pred,1)
        y_pred_soft_max = F.softmax(y_pred,dim=1)
        y_pred_soft_max_1_active = y_pred_soft_max[:,1]
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size, prog_bar=True)

        output = self.test_metrics(y_pred_soft_max_1_active,label.int())
        self.log_dict(output,on_step=False,on_epoch=True,batch_size=self.hparams.batch_size)
        self.test_metrics.update(y_pred_soft_max_1_active,label.int())

        comp_id_pd = pd.Series(comp_id,name="comp_id")
        label_pd = pd.Series(label.cpu(),name="label")
        pred_pd = pd.Series(preds.cpu(),name="prediction")
        pred_0_pd = pd.Series(y_pred_soft_max[:,0].cpu(),name="0_inactive_probability")
        pred_1_pd = pd.Series(y_pred_soft_max[:,1].cpu(),name="1_active_probability")
        batch_predictions = pd.concat([comp_id_pd,label_pd,pred_pd,pred_0_pd,pred_1_pd],axis=1)
        self.test_predictions = pd.concat([self.test_predictions,batch_predictions],axis=0)

    def on_test_end(self):
        self.test_predictions["abs_prob_diff"] =  self.test_predictions["0_inactive_probability"] - self.test_predictions["1_active_probability"]
        self.test_predictions["abs_prob_diff"] = abs(self.test_predictions["abs_prob_diff"])
        self.test_predictions.to_csv(os.path.join(self.hparams.experiment_result_path,f"test_{self.hparams.target}_{self.hparams.fully_layer_1}-{self.hparams.fully_layer_2}-{self.hparams.learning_rate}-{self.hparams.drop_rate}-{self.hparams.batch_size}.csv"),index=False)
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img_arrs, comp_id = batch
        y_pred = self.forward(img_arrs)
        _, preds = torch.max(y_pred,1)
        comp_id_pd = pd.Series(comp_id,name="comp_id")
        pred_pd = pd.Series(preds.cpu(),name="prediction")
        y_pred_soft_max = F.softmax(y_pred,dim=1) 
        pred_0_pd = pd.Series(y_pred_soft_max[:,0].cpu(),name="0_inactive_probability")
        pred_1_pd = pd.Series(y_pred_soft_max[:,1].cpu(),name="1_active_probability")
        batch_predictions = pd.concat([comp_id_pd,pred_pd,pred_0_pd,pred_1_pd],axis=1)
        self.predictions = pd.concat([self.predictions,batch_predictions],axis=0)
    
    def on_predict_end(self):
        self.predictions["abs_prob_diff"] =  self.predictions["0_inactive_probability"] - self.predictions["1_active_probability"]
        self.predictions["abs_prob_diff"] = abs(self.predictions["abs_prob_diff"])
        self.predictions.to_csv(os.path.join(self.hparams.experiment_result_path,f"predictions_{self.hparams.target}_{self.hparams.fully_layer_1}-{self.hparams.fully_layer_2}-{self.hparams.learning_rate}-{self.hparams.drop_rate}-{self.hparams.batch_size}.csv"),index=False)