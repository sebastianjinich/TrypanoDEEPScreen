from rdkit.Chem.Draw import MolDrawOptions
from utils.logging_deepscreen import logger
import multiprocessing
import torch

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
        # Choose to store the images used to train or not
        use_tmp_imgs = False
        return use_tmp_imgs
    
    def get_raytune_scaleing_config(self):
        if configs._get_gpu_number() > 0:
            scaleing_config = {
                "num_workers":1,
                "use_gpu":True,
                "resources_per_worker":{"CPU": self._get_cpu_number(), "GPU": self._get_gpu_number()}
            }
            return scaleing_config
        else:
            scaleing_config = {
                "num_workers":1,
                "use_gpu":False,
            }
            return scaleing_config

    def _get_cpu_number(self):
        cores = multiprocessing.cpu_count() # Count the number of cores in a computer
        gpus = torch.cuda.device_count()
        if gpus*4 <= cores:
            return gpus*4
        else:
            return cores

    def _get_gpu_number(self):
        return torch.cuda.device_count()
        

    
configs = configurations()