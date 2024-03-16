from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
import os
from deepchem.splits import RandomSplitter
from deepchem.data import NumpyDataset
from torchmetrics import classification, MetricCollection
import torch


from utils.configurations import configs
from utils.constants import RANDOM_STATE
from utils.logging_deepscreen import logger
from datasets.datamodule import DEEPscreenDataModule
from engine.system import DEEPScreenClassifier


class deepscreen_ensamble:
    def __init__(self,target,experiments_result_path:str="../../.experiments/",):
        seed_everything(RANDOM_STATE,True)

        self.trainer_models_trained = list()
        self.models_trained = list()
        self.models_ckpt_path = list()
        
        self.target = target
        self.experiment_path_abs = os.path.abspath(experiments_result_path) 
        self.experiment_result_path = os.path.join(self.experiment_path_abs,target,"ensamble")


    def load_models(self,ensamble_checkpoints_directory:str):
        '''
        Loads models stored in lightning checkpoints
        '''
        models_ckpt_paths = [os.path.join(ensamble_checkpoints_directory,path) for path in os.listdir(ensamble_checkpoints_directory) if path.find(".ckpt") != -1]
        self.models_trained = [DEEPScreenClassifier.load_from_checkpoint(model_path) for model_path in models_ckpt_paths]
        self.hyperparameters = self.models_trained[0].hparams

    
    def fit(self,data,hyperparameters,number_to_ensamble:int,max_epochs:int,metric_to_optimize:str,optimize_mode:str):
        '''
        train models, filling self.trainter_models list and self.models_ckpt_path. Then is stores the ckpt path list as a csv/json in the expermient folder
        '''

        datasets = self._generate_n_datasets_random(data,number_to_ensamble)

        self.hyperparameters = hyperparameters 

        for i, dataset in enumerate(datasets):
            tb_logger = TensorBoardLogger(save_dir=self.experiment_result_path)
            checkpoint_callback = ModelCheckpoint(dirpath=self.experiment_result_path, save_top_k=1, monitor=metric_to_optimize, mode=optimize_mode)
            trainer = Trainer(max_epochs=max_epochs,callbacks=[checkpoint_callback],logger=tb_logger)
            model = DEEPScreenClassifier(**hyperparameters,experiment_result_path=self.experiment_result_path,target=self.target)
            train_count = dataset[dataset.data_split == "train"]["bioactivity"].value_counts()
            validation_count = dataset[dataset.data_split == "validation"]["bioactivity"].value_counts()
            logger.info(f"trining ensamble model with dataset train: {train_count[0]+train_count[1]}|{train_count[0]}/{train_count[1]} - validation: {validation_count[0]+validation_count[1]}|{validation_count[0]}/{validation_count[1]}")
            datamodule = DEEPscreenDataModule(data=dataset,batch_size=hyperparameters["batch_size"],experiment_result_path=self.experiment_result_path,data_split_mode="non_random_split",tmp_imgs=False)
            trainer.fit(model,datamodule=datamodule)
            self.trainer_models_trained.append(trainer)
            self.models_ckpt_path.append(checkpoint_callback.best_model_path)
            self.models_trained.append(model)
            logger.info(f"Ensamble model trained {i}/{len(datasets)} - path {checkpoint_callback.best_model_path}")

        return self.trainer_models_trained

    def test(self,data, pred_column = "ensambled_deepscreen_score", label_column = "label"):
        
        prediction =  self.predict(data)

        metrics = MetricCollection(
             {
                "acc": classification.BinaryAccuracy(threshold=0.5),
                "prec": classification.BinaryPrecision(threshold=0.5),
                "f1": classification.BinaryF1Score(threshold=0.5),
                "mcc": classification.BinaryMatthewsCorrCoef(threshold=0.5),
                "recall": classification.BinaryRecall(threshold=0.5),
                "auroc": classification.BinaryAUROC(),
                "auroc_15": classification.BinaryAUROC(max_fpr=0.15),
                "calibration_error": classification.BinaryCalibrationError()
             }
         )
        
        m = metrics(torch.tensor(prediction[pred_column]),torch.tensor(prediction[label_column]))

        for k,v in m.items():
            m[k] = float(v)
        
        return m, prediction

    
    def predict(self,data):

        data_input = data.copy()

        self._check_models_aviable()

        datamodule = DEEPscreenDataModule(data=data_input,batch_size=self.hyperparameters["batch_size"],experiment_result_path=self.experiment_result_path,data_split_mode="predict",tmp_imgs=False)
        
        ensambled_prob_columns_name = list()

        for i,model in enumerate(self.models_trained):
            trainer = Trainer()
            predictions_batches = trainer.predict(model=model,datamodule=datamodule)
            predictions = pd.concat(predictions_batches)
            predictions = predictions[["comp_id","1_active_probability"]]
            prediction_column_name = f"prediction_{i}"
            predictions = predictions.rename(columns={"1_active_probability":prediction_column_name})
            ensambled_prob_columns_name.append(prediction_column_name)
            data_input = pd.merge(data_input,predictions,on="comp_id")

        data_input["ensambled_deepscreen_score"] = data_input.apply(lambda x: x[ensambled_prob_columns_name].mean(),axis=1)
        data_input["ensambled_prediction"] = data_input["ensambled_deepscreen_score"].round().astype(int)

        return data_input
    
    def _generate_n_datasets_random(self,data,number_datasets)->list:

        splitter = RandomSplitter()
        datas = list()
        for i in range(number_datasets):
            comp_id = data.comp_id.to_numpy()
            label_bioact = data.bioactivity.to_numpy()
            dc_dataset_split = NumpyDataset(X=comp_id,y=label_bioact,ids=comp_id)
            train,valid,test = splitter.train_valid_test_split(dc_dataset_split,seed=i,**configs.get_datas_splitting_config())
            data.loc[data.comp_id.isin(test.ids),"data_split"] = "test"
            data.loc[data.comp_id.isin(train.ids),"data_split"] = "train"
            data.loc[data.comp_id.isin(valid.ids),"data_split"] = "validation"
            datas.append(data.copy())

        return datas

    def _check_models_aviable(self):
        if len(self.models_trained) < 0:
            logger.error("Theres not any trained model to predict/test in ensamble. Load or train models")
            raise RuntimeError("Missing models in ensamble training")
        
