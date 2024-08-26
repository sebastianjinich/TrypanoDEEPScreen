from find_max_similiarity import find_max_similiarity
from scipy.special import softmax
import pandas as pd


def tanimoto_bioactivity_predictor(dataset_train:pd.DataFrame,dataset_predict:pd.DataFrame):
    predict_output = dataset_predict.copy()
    predict_output = predict_output.reset_index(drop=True)

    train_positives = dataset_train[dataset_train.bioactivity == 1]
    train_negatives = dataset_train[dataset_train.bioactivity == 0]

    predict_output["positive_comps_tanimoto"] = find_max_similiarity(predict_output,train_positives)["best_tanimoto"]
    predict_output["negative_comps_tanimoto"] = find_max_similiarity(predict_output,train_negatives)["best_tanimoto"]

    predict_output[["1_active_probability","0_inactive_probability"]] = softmax(predict_output[["positive_comps_tanimoto","negative_comps_tanimoto"]],axis=1)

    return predict_output