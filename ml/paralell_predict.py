import ray
from engine.hyperparameters_tune_raytune import deepscreen_hyperparameter_tuneing
from utils.configurations import configs
from engine.system import DEEPScreenClassifier
from datasets.datamodule import DEEPscreenDataModule
from utils.logging_deepscreen import logger

import pandas as pd
import os
import torch
from lightning import Trainer
import glob
from ray.tune import ExperimentAnalysis
from tempfile import TemporaryDirectory


experiment_result_path = "/home/sjinich/disco/TrypanoDEEPscreen/experiments/chembl221_34_auroc_max"
data_input = "/home/sjinich/disco/TrypanoDEEPscreen/data/processed/CHEMBL221_chemblv34.csv"
metric = "val_auroc"


# Función para ejecutar predicciones con un modelo y un datamodule
@ray.remote
def predict_model(model_ckpt_path, data, directory, model_number):
    model = DEEPScreenClassifier.load_from_checkpoint(model_ckpt_path, experiment_result_path=experiment_result_path, batch_size=64)
    datamodule = DEEPscreenDataModule(data=data, batch_size=64, experiment_result_path=experiment_result_path, data_split_mode="predict", tmp_imgs=True)


    trainer = Trainer(
        devices=1, 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=1,
        enable_progress_bar=False,
    )

    predictions = trainer.predict(model, datamodule=datamodule)
    predictions = pd.concat(predictions)
    predictions = predictions.reset_index(drop=True)

    model_name = f"model_{model_number}"
    predictions = predictions.rename(columns={"1_active_probability": model_name})

    output = predictions[["comp_id", model_name]]
    output.to_csv(os.path.join(directory, f"prediction_model_{model_number}.csv"))


def get_result_df(experiment_path):
    experiment = ExperimentAnalysis(experiment_checkpoint_path=experiment_path)
    return experiment.results_df

def main():
    ray.init(num_cpus=20)  # Inicia Ray y detecta recursos disponibles

    ensambe_model_path = "/home/sjinich/disco/TrypanoDEEPscreen/analysis/ensamble_models"
    tmp_dir = TemporaryDirectory()

    data = pd.read_csv(data_input)
    data = data[data.data_split == "test"]
    path = experiment_result_path

    result_df = get_result_df(path)
    result_df = result_df.reset_index()

    top_n_models = 20  # Cambia según la cantidad de modelos a probar
    trials = result_df.sort_values(metric, ascending=False).head(top_n_models)["trial_id"].to_list()
    checkpoint_paths = []
    for trial in trials:
        checkpoint_paths += glob.glob(os.path.join(path, f"*{trial}*/checkpoint*/*.ckpt"))

    # Ejecutar predicciones en paralelo con Ray
    tasks = [
        predict_model.remote(hparams, data, tmp_dir.name, i)
        for i, hparams in enumerate(checkpoint_paths)
    ]
    ray.get(tasks)  # Esperar que todas las tareas terminen

    # Concatenar resultados y verificar unicidad de comp_id
    preds = [pd.read_csv(os.path.join(tmp_dir.name, f), index_col=0) for f in os.listdir(tmp_dir.name)]
    for df in preds:
        if not df["comp_id"].is_unique:
            raise ValueError("Identificador único 'comp_id' duplicado en predicciones.")

    output = pd.concat(preds, axis=1)
    output = output.loc[:, ~output.columns.duplicated()]
    output = output.sort_index(axis=1)  # Ordenar columnas alfabéticamente (model_1, model_2, ...)

    tmp_dir.cleanup()
    ray.shutdown()  # Cierra Ray
    return output


if __name__ == "__main__":
    final_output = main()
    print(final_output)
