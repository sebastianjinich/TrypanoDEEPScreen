import pandas as pd
from engine.hyperparameters_tune_raytune import deepscreen_hyperparameter_tuneing
from utils.configurations import configs
import os
from engine.system import DEEPScreenClassifier
from datasets.datamodule import DEEPscreenDataModule
from lightning import Trainer
import numpy as np
import math
import json
import pickle
from pathlib import Path


def main_raytune_search_train(data_train_val_test:str, target_name_experiment:str, data_split_mode:str, experiment_result_path:str):
    
    data = pd.read_csv(data_train_val_test)
    
    # Performing the whole hyperparameter search with ray tune

    search_space_deepscreen = configs.get_hyperparameters_search()

    hyperparameters_search_setup = configs.get_hyperparameters_search_setup()
    
    tuner = deepscreen_hyperparameter_tuneing(
        data=data,
        data_split_mode=data_split_mode,
        search_space=search_space_deepscreen,
        target=target_name_experiment,
        experiments_result_path=experiment_result_path,
        **hyperparameters_search_setup
        )
    
    result = tuner.tune_deepscreen()
    result_df = result.get_dataframe()
    result_df_path = os.path.join(experiment_result_path,f"raytune_result_{target_name_experiment}.pkl")
    result_df.to_pickle(result_df_path)

    # Performing the temperature search and testing

    best_result = result.get_best_result(metric=hyperparameters_search_setup["metric_to_optimize"],mode=hyperparameters_search_setup["optimize_mode"])
    best_checkpoint_path = best_result.checkpoint.path
    best_checkpoint_path_ckpt = os.path.join(best_checkpoint_path,"checkpoint.ckpt")
    best_result_hparams = best_result.config["train_loop_config"]

    datamodule = DEEPscreenDataModule(data=data,
                                      target_id=target_name_experiment,
                                      batch_size=best_result_hparams["batch_size"],
                                      experiment_result_path=experiment_result_path,
                                      data_split_mode=data_split_mode,
                                      tmp_imgs=True)
    
    # Temperature search and updateing the stored hyperparameters
    
    best_temperature = temperature_scaleing_search(best_checkpoint_path_ckpt,experiment_result_path,datamodule)

    # Testing
    trainer = Trainer()
    model = DEEPScreenClassifier.load_from_checkpoint(best_checkpoint_path_ckpt,experiment_result_path=experiment_result_path,temperature_scaleing=best_temperature)
    trainer.test(model,datamodule=datamodule)



def temperature_scaleing_search(model_checkpoint_path_ckpt,experiment_result_path,datamodule):
    
    best_temperature = None
    best_calibration_error = math.inf

    for temp in configs.get_temperature_search_configs():

        trainer = Trainer()

        model = DEEPScreenClassifier.load_from_checkpoint(model_checkpoint_path_ckpt,experiment_result_path=experiment_result_path,temperature_scaleing=temp)

        val_results = trainer.validate(model,datamodule=datamodule,verbose=False)

        if val_results[0]["val_calibration_error"] < best_calibration_error:
            best_calibration_error = val_results[0]["val_calibration_error"]
            best_temperature = temp
            print("new best temp:",best_temperature)
    
    return best_temperature
