from engine.ensamble_model_train import deepscreen_ensamble
import pandas as pd

ensamble = deepscreen_ensamble("antibioticos")

antibioticos_train = pd.read_csv("../../.data/processed/antibiotics_random_split.csv")
antibioticos_test = pd.read_csv("../../.data/processed/antibiotics_test_predict_only_assayed.csv")

hpams = {
    "fully_layer_1": 32, "fully_layer_2": 256, "drop_rate": 0.3, "learning_rate": 0.001, "batch_size": 32
}

ensambled_models_paths = ensamble.fit(data=antibioticos_train,hyperparameters=hpams,number_to_ensamble=20,max_epochs=200,metric_to_optimize="val_mcc",optimize_mode="max")

print(f"Ensamled models trained: {ensambled_models_paths}")

metrics, prediction_test = ensamble.test(antibioticos_test)

print(f"Testing results \n {metrics}")

prediction_test.to_csv("../../.experiments/antibioticos/testing_results_antibioticos.csv",index=False)