from rdkit.Chem.Draw import MolDrawOptions
from logging_deepscreen import logger


class configurations:
    def __init__(self):
        logger.debug("Configuration class instansiated")

        # default (arbitrary) hyperparameters
        self.hyperparameters = {
        'fully_layer_1': 256,
        'fully_layer_2': 32,
        'learning_rate': 0.001,
        'batch_size': 32,
        'drop_rate': 0.5,
        'n_epoch': 200
        }

    def get_mol_draw_options(self):
        opt = MolDrawOptions()
        opt.atomLabelFontSize = 55
        opt.dotsPerAngstrom = 100
        opt.bondLineWidth = 1
        return opt
    
    def get_img_size(self):
        img_size = 200
        return (img_size,img_size)
    
    def get_hyperparameters(self):
        
        return self.hyperparameters
    
    def set_hyperparameters(self,new_hyperparamenters:dict):
        self.hyperparameters = new_hyperparamenters
        logger.info(f"Hyperparameters witched to {self.hyperparameters}")

        #TODO santity seted checking hyperparameters
        return self.hyperparameters

    
configs = configurations()