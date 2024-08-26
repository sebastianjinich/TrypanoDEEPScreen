import pandas as pd
from engine.hyperparameters_tune_raytune_rdkit_features import deepscreen_hyperparameter_tuneing
from utils.configurations import configs
import os
from engine.system_rdkit_features import DEEPScreenClassifier
from datasets.datamodule_rdkit_features import DEEPscreenDataModule
from lightning import Trainer
from utils.logging_deepscreen import logger
import numpy as np


def main_raytune_search_train(data_train_val_test:str, features_rdkit:str,target_name_experiment:str, data_split_mode:str, experiment_result_path:str):
    
    data = pd.read_csv(data_train_val_test)
    features = np.load(features_rdkit)

    features_len = len(features[0])
    
    search_space_deepscreen = configs.get_hyperparameters_search()

    hyperparameters_search_setup = configs.get_hyperparameters_search_setup()
    
    tuner = deepscreen_hyperparameter_tuneing(
        data=data,
        features=features,
        data_split_mode=data_split_mode,
        search_space=search_space_deepscreen,
        target=target_name_experiment,
        experiments_result_path=experiment_result_path,
        **hyperparameters_search_setup
        )
    
    result = tuner.tune_deepscreen()

    best_result = result.get_best_result(metric=hyperparameters_search_setup["metric_to_optimize"],mode=hyperparameters_search_setup["optimize_mode"])
    best_checkpoint_path = os.path.join(best_result.checkpoint.path,"checkpoint.ckpt")
    best_result_hparams = best_result.config["train_loop_config"]
    best_val_metrics = {i:j for i,j in best_result.metrics.items() if i.find("val") != -1 }
    logger.info(f"Best hyperparams founded {best_result_hparams}")
    logger.info(f"Best validation metrics {best_val_metrics}")

    result_df = result.get_dataframe()
    result_df_path = os.path.join(experiment_result_path,f"raytune_result_{target_name_experiment}.pkl")
    result_df.to_pickle(result_df_path)

    trainer = Trainer()
    model = DEEPScreenClassifier.load_from_checkpoint(best_checkpoint_path,experiment_result_path=experiment_result_path)
    datamodule = DEEPscreenDataModule(data=data,
                                      features=features,
                                      batch_size=best_result_hparams["batch_size"],
                                      experiment_result_path=experiment_result_path,
                                      data_split_mode=data_split_mode,
                                      tmp_imgs=False)
    
    trainer.test(model,datamodule=datamodule)

