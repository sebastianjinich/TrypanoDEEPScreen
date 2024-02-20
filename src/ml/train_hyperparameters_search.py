import pandas as pd
from ray import tune
from engine.hyperparameters_tune_raytune import deepscreen_hyperparameter_tuneing

data = pd.read_csv("../../.data/processed/CHEMBL2581.csv")
search_space_deepscreen = {
        'fully_layer_1': tune.choice([16, 32, 128, 256, 512]),
        'fully_layer_2': tune.choice([16, 32, 128, 256, 512]),
        'learning_rate': tune.choice([0.0005, 0.0001, 0.005, 0.001, 0.01]),
        'batch_size': tune.choice([32, 64]),
        'drop_rate': tune.choice([0.2, 0.3, 0.5, 0.6]),
    }
tuner = deepscreen_hyperparameter_tuneing(data,search_space_deepscreen,"CHEMBL2581",num_samples=50)
result = tuner.tune_deepscreen()
result_df = result.get_dataframe()
result_df.to_pickle("../../.experiments/raytune_asha_chembl2581.pkl")