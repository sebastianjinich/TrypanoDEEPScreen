from rdkit.Chem.Draw import MolDrawOptions
from utils.logging_deepscreen import logger
import multiprocessing
import torch
from ray import tune
import math

class configurations:
    def __init__(self):
        logger.debug("Configuration class instansiated")

    def get_mol_draw_options(self):
        opt = MolDrawOptions()
        opt.atomLabelFontSize = 55
        opt.dotsPerAngstrom = 100
        opt.bondLineWidth = 1
        return opt
    
    def get_img_size(self):
        img_size = 200
        return (img_size,img_size)

    def get_use_tmp_imgs(self):
        # Choose to store the images used to train or not (False = Store images)
        use_tmp_imgs = False
        return use_tmp_imgs
    
    def get_hyperparameters_search(self):
        hyperparams_choices = {
        'fully_layer_1': tune.choice([16, 32, 128, 256, 512]),
        'fully_layer_2': tune.choice([16, 32, 128, 256, 512]),
        'learning_rate': tune.choice([0.0005, 0.0001, 0.005, 0.001, 0.01]),
        'batch_size': tune.choice([32, 64]),
        'drop_rate': tune.choice([0.3, 0.5, 0.6, 0.8]),
        }
        return hyperparams_choices
    
    def get_hyperparameters_search_setup(self):
        setup = {
           "max_epochs": 200,
           "grace_period": 55,
           "metric_to_optimize": "val_mcc",
           "optimize_mode":"max",
           "num_samples":320,
           "asha_reduction_factor":3,
           "number_ckpts_keep":1
        }
        return setup

    def get_raytune_scaleing_config(self):

        if self._get_gpu_number() > 0:
            scaleing_config = {
                "num_workers":self._get_gpu_number(),
                "use_gpu":True,
                "resources_per_worker":{"CPU": self._get_cpu_number(), "GPU": 1}
            }
            return scaleing_config
        else:
            scaleing_config = {
                "num_workers":1,
                "use_gpu":False,
            }
            return scaleing_config
    
    def get_datas_splitting_config(self):
        # configuration of deepchem splitting object.
        config = {
            "frac_train":0.8,
            "frac_valid":0.2,
            "frac_test":0
        }
        return config
        
    def _get_cpu_number(self):
        cores = multiprocessing.cpu_count()
        gpus = self._get_gpu_number()

        if self._get_gpu_number() > 0:
            if gpus*4 <= cores:
                return gpus*4
            else:
                return math.floor(cores/gpus)

    def _get_gpu_number(self):
        return torch.cuda.device_count()
        

    
configs = configurations()