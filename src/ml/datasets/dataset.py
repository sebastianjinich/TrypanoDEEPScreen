import os
import pandas as pd
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
from rdkit.Chem import Draw, MolFromSmiles
from ..utils.constants import RANDOM_STATE, DEFAULT_IMG_SIZE, DEFAULT_ATOM_LABEL_FONTSIZE, DEFAULT_BOND_LINE_WIDTH, DEFAULT_DOTS_PER_ANGSTROM
from ..utils.logging_deepscreen import logger

random.seed(RANDOM_STATE)

class DEEPScreenDataset(Dataset):
    def __init__(self, path_tmp_files:str, df_compid_smiles_bioactivity:pd.DataFrame, smiles_column = 'smiles', compound_id_column = 'comp_id', bioactivity_label_column = 'bioactivity_label'):
            self.path = path_tmp_files
            self.path_imgs = os.path.join(self.path,'imgs')
            self.df = df_compid_smiles_bioactivity.copy()
            self.compid_column = compound_id_column
            self.smiles_column = smiles_column
            self.label_column  = bioactivity_label_column
            if not os.path.exists(self.path_imgs):
                os.makedirs(self.path_imgs)

            # creating molecules images -> path will be stored in 'img_molecule' column
            self.df['img_molecule'] = self.df.apply(lambda x: self.smiles_to_img_png(x[compound_id_column],x[smiles_column],self.path_imgs),axis=1)
            logger.debug(f'Dataset created in {path_tmp_files}')

    def __len__(self):
        return len(self.df.index)
    
    def smiles_to_img_png(self, comp_id:str, smiles:str, output_path:str, IMG_SIZE:int=DEFAULT_IMG_SIZE, atomLabelFontSize:int=DEFAULT_ATOM_LABEL_FONTSIZE, dotsPerAngstrom:int=DEFAULT_DOTS_PER_ANGSTROM, bondLineWidth:int=DEFAULT_BOND_LINE_WIDTH)->str:
        '''
        Given an id and an output path the function will create a image with the 2D molecule. 

        Returns: the path to the file i.e. /output_path/comp_id.png
        '''
        mol = MolFromSmiles(smiles)
        opt = Draw.MolDrawOptions()
        opt.atomLabelFontSize = atomLabelFontSize
        opt.dotsPerAngstrom = dotsPerAngstrom
        opt.bondLineWidth = bondLineWidth
        output_file = os.path.join(output_path, f"{comp_id}.png")
        Draw.MolToFile(mol, output_file, size= (IMG_SIZE, IMG_SIZE), options=opt)
        return output_file


class DEEPScreenDatasetPredict(DEEPScreenDataset):
    def __init__(self, path_tmp_files:str, df_compid_smiles_bioactivity:pd.DataFrame, smiles_column = 'smiles', compound_id_column = 'comp_id', bioactivity_label_column = 'bioactivity_label'):
            self.path = path_tmp_files
            self.path_imgs = os.path.join(self.path,'imgs')
            self.df = df_compid_smiles_bioactivity.copy()
            self.compid_column = compound_id_column
            self.smiles_column = smiles_column
            self.label_column  = bioactivity_label_column
            if not os.path.exists(self.path_imgs):
                os.makedirs(self.path_imgs)

            # creating molecules images -> path will be stored in 'img_molecule' column
            self.df['img_molecule'] = self.df.apply(lambda x: self.smiles_to_img_png(x[compound_id_column],x[smiles_column],self.path_imgs),axis=1)
            logger.debug(f'Dataset created in {path_tmp_files}')


    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row[self.compid_column]
        img_path = row['img_molecule']
        img_arr = cv2.imread(img_path)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        return img_arr, comp_id

class DEEPScreenDatasetTrain(DEEPScreenDataset):
    def __init__(self, path_tmp_files:str, df_compid_smiles_bioactivity:pd.DataFrame, smiles_column = 'smiles', compound_id_column = 'comp_id', bioactivity_label_column = 'bioactivity_label'):
            self.path = path_tmp_files
            self.path_imgs = os.path.join(self.path,'imgs')
            self.df = df_compid_smiles_bioactivity.copy()
            self.compid_column = compound_id_column
            self.smiles_column = smiles_column
            self.label_column  = bioactivity_label_column
            if not os.path.exists(self.path_imgs):
                os.makedirs(self.path_imgs)

            # creating molecules images -> path will be stored in 'img_molecule' column
            self.df['img_molecule'] = self.df.apply(lambda x: self.smiles_to_img_png(x[compound_id_column],x[smiles_column],self.path_imgs),axis=1)
            logger.debug(f'Dataset created in {path_tmp_files}')

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row[self.compid_column]
        img_path = row['img_molecule']
        img_arr = cv2.imread(img_path)
        if random.random()>=0.50:
            angle = random.randint(0,359)
            rows, cols, channel = img_arr.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_arr = cv2.warpAffine(img_arr, rotation_matrix, (cols, rows), cv2.INTER_LINEAR,
                                             borderValue=(255, 255, 255))  # cv2.BORDER_CONSTANT, 255)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        label = row[self.label_column]

        return img_arr, label, comp_id
    
    