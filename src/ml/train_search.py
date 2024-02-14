import pandas as pd
from ray import tune
from scripts.train_raytune import deepscreen_hyperparameter_tuneing

data = pd.read_csv("../../.data/processed/CHEMBL5567.csv")
search_space_deepscreen = {
        'fully_layer_1': tune.choice([16, 32, 128, 256, 512]),
        'fully_layer_2': tune.choice([16, 32, 128, 256, 512]),
        'learning_rate': tune.choice([0.0005, 0.0001, 0.005, 0.001, 0.01]),
        'batch_size': tune.choice([32, 64]),
        'drop_rate': tune.choice([0.5, 0.6, 0.8]),
    }
tuner = deepscreen_hyperparameter_tuneing(data,search_space_deepscreen,"chembl5567",num_samples=300)
result = tuner.tune_deepscreen()