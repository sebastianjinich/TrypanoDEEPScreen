from rdkit.Chem.Draw import MolDrawOptions
from utils.logging_deepscreen import logger


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
    

    
configs = configurations()