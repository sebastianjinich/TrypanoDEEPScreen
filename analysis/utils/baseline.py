import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
import warnings
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import statistics
import sys; sys.path.insert(0, '/home/sjinich/disco/jcompoundmapper_pywrapper/src')
from jCompoundMapper_pywrapper import JCompoundMapper


# PREDICTORES

def similarity_softmax_bioactivity_predictor(dataset_train:pd.DataFrame,dataset_predict:pd.DataFrame,similarity_scorer,fps_generator,similarity_function:DataStructs):
    predict_output = dataset_predict.copy()
    predict_output = predict_output.reset_index(drop=True)

    train_positives = dataset_train[dataset_train.bioactivity == 1]
    train_negatives = dataset_train[dataset_train.bioactivity == 0]

    predict_output["positive_comps_similarity"] = calculate_similiarity_scores(predict_output,train_positives,similarity_scorer,fps_generator,similarity_function)["pred_score"]
    predict_output["negative_comps_similarity"] = calculate_similiarity_scores(predict_output,train_negatives,similarity_scorer,fps_generator,similarity_function)["pred_score"]

    predict_output[["prediction_score","negative_prediction_score"]] = softmax(predict_output[["positive_comps_similarity","negative_comps_similarity"]],axis=1)

    return predict_output

def similarity_bioactivity_predictor(dataset_train:pd.DataFrame,dataset_predict:pd.DataFrame,similarity_scorer,fps_generator,similarity_function:DataStructs):
    predict_output = dataset_predict.copy()
    predict_output = predict_output.reset_index(drop=True)

    predict_output["prediction_score"] = calculate_similiarity_scores(predict_output,dataset_train,similarity_scorer,fps_generator,similarity_function)["pred_score"]

    return predict_output

# GENERADORES DE FINGERPRINTS

def fingerprint_generator_rdkit(df):
    df["ROMol"] = df.loc[:,"smiles"].apply(Chem.MolFromSmiles)
    df["fps"] = df.loc[:,"ROMol"].apply(FingerprintMols.FingerprintMol)
    return df

def fingerprint_generator_ASP(df): 
    jcm = JCompoundMapper("ASP")
    df["ROMol"] = df.loc[:,"smiles"].apply(Chem.MolFromSmiles)
    fingerprints = jcm.calculate(df.loc[:,"ROMol"])
    for idx in fingerprints.index:
        df.loc[idx,"fps"] = list_to_sparse_int_vect(fingerprints.loc[idx,:])
    return df

def fingerprint_generator_LSTAR(df): 
    jcm = JCompoundMapper("LSTAR")
    df["ROMol"] = df.loc[:,"smiles"].apply(Chem.MolFromSmiles)
    fingerprints = jcm.calculate(df.loc[:,"ROMol"])
    for idx in fingerprints.index:
        df.loc[idx,"fps"] = list_to_sparse_int_vect(fingerprints.loc[idx,:])
    return df

def fingerprint_generator_RAD2D(df): 
    jcm = JCompoundMapper("RAD2D")
    df["ROMol"] = df.loc[:,"smiles"].apply(Chem.MolFromSmiles)
    fingerprints = jcm.calculate(df.loc[:,"ROMol"])
    for idx in fingerprints.index:
        df.loc[idx,"fps"] = list_to_sparse_int_vect(fingerprints.loc[idx,:])
    return df

def list_to_sparse_int_vect(fp_list):
    from rdkit import DataStructs
    from rdkit.DataStructs.cDataStructs import SparseBitVect
    # Crear un SparseIntVect con el tamaño adecuado
    sparse_vect = SparseBitVect(len(fp_list))

    # Añadir los bits activados
    for idx, value in enumerate(fp_list):
        sparse_vect[idx] = int(value)

    return sparse_vect



# SCORERS
# Son los que apartir de todas las similitudes integran un valor unico

def max_similarity_scorer(fps_to_compare, all_fps_to_compare_with, labels, similarity:DataStructs):
    all_comparisons = [similarity(fps_to_compare, fps) for fps in all_fps_to_compare_with]
    return max(all_comparisons)

def mean_similarity_scorer(fps_to_compare, all_fps_to_compare_with, labels, similarity:DataStructs):
    all_comparisons = [similarity(fps_to_compare, fps) for fps in all_fps_to_compare_with]
    return statistics.mean(all_comparisons)

def auroc_scorer(fps_to_compare, all_fps_to_compare_with, labels, similarity:DataStructs):
    all_comparisons = [similarity(fps_to_compare, fps) for fps in all_fps_to_compare_with]
    return roc_auc_score(labels,all_comparisons)

def aupr_scorer(fps_to_compare, all_fps_to_compare_with, labels, similarity:DataStructs):
    all_comparisons = [similarity(fps_to_compare, fps) for fps in all_fps_to_compare_with]
    return average_precision_score(labels,all_comparisons)

def upper_cuartile_scorer(fps_to_compare, all_fps_to_compare_with, labels, similarity:DataStructs):
    all_comparisons = [similarity(fps_to_compare, fps) for fps in all_fps_to_compare_with]
    arr = np.array(all_comparisons)
    upper_quartile_value = np.percentile(arr, 75)
    upper_quartile_elements = arr[arr >= upper_quartile_value]
    average = np.mean(upper_quartile_elements)
    return average

def upper_decile_scorer(fps_to_compare, all_fps_to_compare_with, labels, similarity:DataStructs):
    all_comparisons = [similarity(fps_to_compare, fps) for fps in all_fps_to_compare_with]
    arr = np.array(all_comparisons)
    upper_quartile_value = np.percentile(arr, 90)
    upper_quartile_elements = arr[arr >= upper_quartile_value]
    average = np.mean(upper_quartile_elements)
    return average

# Funcion auxiliar

def calculate_similiarity_scores(similarity_of:pd.DataFrame, similarity_to:pd.DataFrame, scorer, fingerprinter, similarity_func, smiles_column = "smiles"):

    similarity_of_internal = similarity_of.copy()

    similarity_to_internal = similarity_to.copy()


    # Configurar el manejo de advertencias
    warnings.filterwarnings('ignore')

    similarity_of_internal.loc[:,"pred_score"] = similarity_of_internal.loc[:,"fps"].apply(lambda x: scorer(x,similarity_to_internal["fps"],labels=similarity_to_internal["bioactivity"], similarity = similarity_func))
    
    return similarity_of_internal[similarity_of.columns.to_list()+["pred_score"]]
