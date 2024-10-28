# %% [markdown]
# ## Imports

# %%
import pandas as pd
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '../utils')
from utils.baseline import similarity_bioactivity_predictor, similarity_softmax_bioactivity_predictor, aupr_scorer, max_similarity_scorer, auroc_scorer, mean_similarity_scorer, fingerprint_generator_rdkit, fingerprint_generator_ASP, fingerprint_generator_LSTAR, fingerprint_generator_RAD2D, upper_cuartile_scorer, upper_decile_scorer
from rdkit import DataStructs
import sys; sys.path.insert(0, '/home/sjinich/disco/jcompoundmapper_pywrapper/src')
from jCompoundMapper_pywrapper import JCompoundMapper
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# %% [markdown]
# ## Baseline

# %%
fingerprinters = {
    "RDKIT":fingerprint_generator_rdkit,
    "LSTAR":fingerprint_generator_LSTAR,
    "ASP":fingerprint_generator_ASP,
    "RAD2D":fingerprint_generator_RAD2D
}

similarity_functions = {
    "BraunBlanquet": DataStructs.BraunBlanquetSimilarity,
    "Tanimoto": DataStructs.TanimotoSimilarity,
    "Cosine": DataStructs.CosineSimilarity
}

# %%
def baseline(target_name, train, validation, fingerprinters, similarity_functions):
    results = {
        f"AUROC_{target_name}":dict(),
        f"AUROC30_{target_name}":dict(),
        f"AUROC15_{target_name}":dict()
    }

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    for title_fps, fps in tqdm(fingerprinters.items(), desc=target_name):
        
        train = fps(train)
        validation = fps(validation)

        for title_similarity, similarity in tqdm(similarity_functions.items(), desc=title_fps):

            prediction_max = similarity_softmax_bioactivity_predictor(train,validation,max_similarity_scorer,fps,similarity)
            prediction_mean = similarity_softmax_bioactivity_predictor(train,validation,mean_similarity_scorer,fps,similarity)
            prediction_cuartile = similarity_softmax_bioactivity_predictor(train,validation,upper_cuartile_scorer,fps,similarity)
            prediction_decile = similarity_softmax_bioactivity_predictor(train,validation,upper_decile_scorer,fps,similarity)
            prediction_auroc = similarity_bioactivity_predictor(train,validation,auroc_scorer,fps,similarity)
            prediction_aupr = similarity_bioactivity_predictor(train,validation,aupr_scorer,fps,similarity)
    
            results[f"AUROC_{target_name}"].update({
                f"max_{title_fps}_{title_similarity}": roc_auc_score(prediction_max["bioactivity"],prediction_max["prediction_score"]),
                f"mean_{title_fps}_{title_similarity}": roc_auc_score(prediction_mean["bioactivity"],prediction_mean["prediction_score"]),
                f"cuartile_{title_fps}_{title_similarity}": roc_auc_score(prediction_cuartile["bioactivity"],prediction_cuartile["prediction_score"]),
                f"decile_{title_fps}_{title_similarity}": roc_auc_score(prediction_decile["bioactivity"],prediction_decile["prediction_score"]),
                f"auroc_{title_fps}_{title_similarity}": roc_auc_score(prediction_auroc["bioactivity"],prediction_auroc["prediction_score"]),
                f"aupr_{title_fps}_{title_similarity}": roc_auc_score(prediction_aupr["bioactivity"],prediction_aupr["prediction_score"]),
            })

            results[f"AUROC30_{target_name}"].update({
                f"max_{title_fps}_{title_similarity}": roc_auc_score(prediction_max["bioactivity"],prediction_max["prediction_score"],max_fpr=0.3),
                f"mean_{title_fps}_{title_similarity}": roc_auc_score(prediction_mean["bioactivity"],prediction_mean["prediction_score"],max_fpr=0.3),
                f"cuartile_{title_fps}_{title_similarity}": roc_auc_score(prediction_cuartile["bioactivity"],prediction_cuartile["prediction_score"],max_fpr=0.3),
                f"decile_{title_fps}_{title_similarity}": roc_auc_score(prediction_decile["bioactivity"],prediction_decile["prediction_score"],max_fpr=0.3),
                f"auroc_{title_fps}_{title_similarity}": roc_auc_score(prediction_auroc["bioactivity"],prediction_auroc["prediction_score"],max_fpr=0.3),
                f"aupr_{title_fps}_{title_similarity}": roc_auc_score(prediction_aupr["bioactivity"],prediction_aupr["prediction_score"],max_fpr=0.3),
            })

            results[f"AUROC15_{target_name}"].update({
                f"max_{title_fps}_{title_similarity}": roc_auc_score(prediction_max["bioactivity"],prediction_max["prediction_score"],max_fpr=0.15),
                f"mean_{title_fps}_{title_similarity}": roc_auc_score(prediction_mean["bioactivity"],prediction_mean["prediction_score"],max_fpr=0.15),
                f"cuartile_{title_fps}_{title_similarity}": roc_auc_score(prediction_cuartile["bioactivity"],prediction_cuartile["prediction_score"],max_fpr=0.15),
                f"decile_{title_fps}_{title_similarity}": roc_auc_score(prediction_decile["bioactivity"],prediction_decile["prediction_score"],max_fpr=0.15),
                f"auroc_{title_fps}_{title_similarity}": roc_auc_score(prediction_auroc["bioactivity"],prediction_auroc["prediction_score"],max_fpr=0.15),
                f"aupr_{title_fps}_{title_similarity}": roc_auc_score(prediction_aupr["bioactivity"],prediction_aupr["prediction_score"],max_fpr=0.15),
            })

    return results

# %%

print("Starting gridsearch")

targets = ["/home/sjinich/disco/TrypanoDEEPscreen/data/processed/CHEMBL262_chemblv34.csv","/home/sjinich/disco/TrypanoDEEPscreen/data/processed/CHEMBL2581_chemblv34.csv","/home/sjinich/disco/TrypanoDEEPscreen/data/processed/CHEMBL2850_chemblv34.csv","/home/sjinich/disco/TrypanoDEEPscreen/data/processed/CHEMBL4072_chemblv34.csv","/home/sjinich/disco/TrypanoDEEPscreen/data/processed/CHEMBL4657_chemblv34.csv","/home/sjinich/disco/TrypanoDEEPscreen/data/processed/CHEMBL5567_chemblv34.csv"]

all_targets_results = pd.DataFrame()

for target in tqdm(targets):
    
    df_all = pd.read_csv(target)

    train = df_all[df_all.data_split == "train"]
    test = df_all[df_all.data_split == "test"]
    validation = df_all[df_all.data_split == "validation"]

    target_name_file = target.split("/")[-1]
    target_name = target_name_file[:target_name_file.find("_chemblv34.csv")]

    result = baseline(target_name,train,test,fingerprinters,similarity_functions)

    all_targets_results = pd.concat([all_targets_results,pd.DataFrame(result)],axis=1)
    

# %%
all_targets_results = all_targets_results.transpose()

# %%
all_targets_results["AUROC"] = all_targets_results.index.to_series()

# %%
all_targets_results["AUROC"] = all_targets_results["AUROC"].apply(lambda x: x.split("_")[0])

# %%
all_targets_results.index = all_targets_results.index.to_series().apply(lambda x: x.split("_")[1])

# %%
all_targets_results.to_csv("baseline_gridsearch_all_targets.csv")

# %%
