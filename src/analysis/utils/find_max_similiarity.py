import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
import warnings



def find_max_similiarity(similarity_of:pd.DataFrame, similarity_to:pd.DataFrame, smiles_column = "smiles"):

    similarity_of_internal = similarity_of.copy()
    similarity_of_internal["ROMol"] = similarity_of_internal.loc[:,smiles_column].apply(Chem.MolFromSmiles)
    similarity_of_internal["fps"] = similarity_of_internal.loc[:,"ROMol"].apply(FingerprintMols.FingerprintMol)
    similarity_of_internal = similarity_of_internal.reset_index(drop=True)

    similarity_to_internal = similarity_to.copy()
    similarity_to_internal["ROMol"] = similarity_to_internal.loc[:,smiles_column].apply(Chem.MolFromSmiles)
    similarity_to_internal["fps"] = similarity_to_internal.loc[:,"ROMol"].apply(FingerprintMols.FingerprintMol)
    similarity_to_internal = similarity_to_internal.reset_index(drop=True)

    # Configurar el manejo de advertencias
    warnings.filterwarnings('ignore')

    similarity_of_internal.loc[:,"best_tanimoto"] = similarity_of_internal.loc[:,"fps"].apply(lambda x: best_tanimoto(x,similarity_to_internal["fps"]))
    
    return similarity_of_internal[similarity_of.columns.to_list()+["best_tanimoto"]]

def best_tanimoto(fps_to_compare,all_fps_to_compare_with):
    all_comparisons = [DataStructs.FingerprintSimilarity(fps_to_compare, fps) for fps in all_fps_to_compare_with]
    return max(all_comparisons)