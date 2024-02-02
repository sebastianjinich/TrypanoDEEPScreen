# External imports
import os
import pandas as pd
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
import torch
from rdkit.Chem import Draw, MolFromSmiles

# Internal Imports
from utils.constants import RANDOM_STATE
from utils.logging_deepscreen import logger
from utils.configurations import configs

random.seed(RANDOM_STATE)

class DEEPScreenDataset(Dataset):
    def __init__(self, path_imgs_files:str, df_compid_smiles_bioactivity:pd.DataFrame):
            super(DEEPScreenDataset, self).__init__()

            self.path_imgs = path_imgs_files
            self.df = df_compid_smiles_bioactivity.copy()
            self.config = configs

            if not os.path.exists(self.path_imgs):
                os.makedirs(self.path_imgs)

            # creating molecules images -> path will be stored in 'img_molecule' column
            self.df['img_molecule'] = self.df.apply(lambda x: self.smiles_to_img_png(x["comp_id"],x["smiles"],self.path_imgs),axis=1)
            logger.debug(f'Dataset created in {self.path_imgs}')


    def __len__(self):
        return len(self.df.index)
    
    def smiles_to_img_png(self, comp_id:str, smiles:str, output_path:str)->str:
        '''
        Given an id and an output path the function will create a image with the 2D molecule. 

        Returns: the path to the file i.e. /output_path/comp_id.png
        '''
        mol = MolFromSmiles(smiles)
        opt = self.config.get_mol_draw_options()
        img_size = self.config.get_img_size()

        output_file = os.path.join(output_path, f"{comp_id}.png")
        Draw.MolToFile(mol, output_file, size=img_size, options=opt)
        return output_file


class DEEPScreenDatasetPredict(DEEPScreenDataset):
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row["comp_id"]
        img_path = row['img_molecule']
        img_arr = cv2.imread(img_path)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        return torch.tensor(img_arr).type(torch.FloatTensor), comp_id
    

class DEEPScreenDatasetTrain(DEEPScreenDataset):

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row["comp_id"]
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
        label = row["bioactivity"]

        return torch.tensor(img_arr).type(torch.FloatTensor), torch.tensor(label), comp_id
    

class DEEPScreenDatasetTest(DEEPScreenDataset):

    def __getitem__(self, index):
        row = self.df.iloc[index]
        comp_id = row["comp_id"]
        img_path = row['img_molecule']
        img_arr = cv2.imread(img_path)
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        label = row["bioactivity"]

        return torch.tensor(img_arr).type(torch.FloatTensor), torch.tensor(label), comp_id