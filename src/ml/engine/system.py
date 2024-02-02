# External imports
import torch
from torch.nn import functional as F
import torch.nn as nn
import lightning as L
from torchmetrics import classification

# Internal imports
from utils.configurations import configs
from utils.logging_deepscreen import logger

class DEEPScreenClassifier(L.LightningModule):
    def __init__(self,fully_layer_1, fully_layer_2, drop_rate, learning_rate, batch_size):
        super(DEEPScreenClassifier, self).__init__()
        self.save_hyperparameters()
        logger.info(f"Training this hyperparameters {[i for i in self.hparams.items()]}") 

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
        self.train_acc = classification.BinaryAccuracy()
        self.val_acc = classification.BinaryAccuracy()

        self.train_pres = classification.BinaryPrecision()
        self.val_pres = classification.BinaryPrecision()

        self.train_f1 = classification.BinaryF1Score()
        self.val_f1 = classification.BinaryF1Score()

        self.train_mcc = classification.BinaryMatthewsCorrCoef()
        self.val_mcc = classification.BinaryMatthewsCorrCoef()

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
        _, preds = torch.max(y_pred,1)
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        
        # performance metrics
        self.train_acc(preds,label)
        self.train_f1(preds,label)
        self.train_mcc(preds,label)
        self.train_pres(preds,label)
        
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        self.log('train_mcc', self.train_mcc, on_step=False, on_epoch=True)
        self.log('train_pres', self.train_pres, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        img_arrs, label, comp_id = val_batch
        y_pred = self.forward(img_arrs)
        _, preds = torch.max(y_pred,1)
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        
        self.val_acc(preds,label)
        self.val_f1(preds,label)
        self.val_mcc(preds,label)
        self.val_pres(preds,label)
        
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val_mcc', self.val_mcc, on_step=False, on_epoch=True)
        self.log('val_pres', self.val_pres, on_step=False, on_epoch=True)

    def test_step(self,test_batch,batch_idx):
        img_arrs, label, comp_id = test_batch
        y_pred = self.forward(img_arrs)
        _, preds = torch.max(y_pred,1)
        loss = self.cross_entropy_loss(y_pred.squeeze(),label)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self.forward(x)