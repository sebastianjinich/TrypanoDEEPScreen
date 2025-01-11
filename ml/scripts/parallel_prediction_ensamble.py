import ray
from engine.system import DEEPScreenClassifier
from datasets.datamodule import DEEPscreenDataModule

import pandas as pd
import os
import torch
from lightning import Trainer
import glob
from ray.tune import ExperimentAnalysis
from tempfile import TemporaryDirectory

# Función para ejecutar predicciones con un modelo y un datamodule
@ray.remote
def predict_model(model_ckpt_path, data, directory, model_number, experiment_result_path):
    model = DEEPScreenClassifier.load_from_checkpoint(model_ckpt_path, experiment_result_path=experiment_result_path, batch_size=64)
    datamodule = DEEPscreenDataModule(data=data, batch_size=64, experiment_result_path=experiment_result_path, data_split_mode="predict", tmp_imgs=True)


    trainer = Trainer(
        devices=1, 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=1,
        enable_progress_bar=False,
        logger=False
    )

    predictions = trainer.predict(model, datamodule=datamodule)
    predictions = pd.concat(predictions)
    predictions = predictions.reset_index(drop=True)

    model_name = f"model_{model_number}"
    predictions = predictions.rename(columns={"1_active_probability": model_name})
    output = predictions
    output.to_csv(os.path.join(directory, f"prediction_model_{model_number}.csv"))


def get_result_df(experiment_path):
    experiment = ExperimentAnalysis(experiment_checkpoint_path=experiment_path)
    return experiment.results_df

def parallel_prediction_ensamble(model_experiments_path,data_input,metric,result_path=None,top_n_hparams=40,n_checkpoints=None):
    ray.init(num_cpus=40)  # Inicia Ray y detecta recursos disponibles

    tmp_dir = TemporaryDirectory()

    data = pd.read_csv(data_input)
    path = model_experiments_path

    result_df = get_result_df(model_experiments_path)
    result_df = result_df.reset_index()

    trials = result_df.sort_values(metric, ascending=False).head(top_n_hparams)["trial_id"].to_list()
    checkpoint_paths = []
    for trial in trials:
        checkpoint_paths += glob.glob(os.path.join(path, f"*{trial}*/checkpoint*/*.ckpt"))

    if checkpoint_paths:
        checkpoint_paths = checkpoint_paths[:n_checkpoints]

    # Ejecutar predicciones en paralelo con Ray
    tasks = [
        predict_model.remote(checkpoint_path, data, tmp_dir.name, i, path) for i, checkpoint_path in enumerate(checkpoint_paths)
    ]
    ray.get(tasks)  # Esperar que todas las tareas terminen

    # Concatenar resultados y verificar unicidad de comp_id
    preds = [pd.read_csv(os.path.join(tmp_dir.name, f), index_col=0) for f in os.listdir(tmp_dir.name)]
    for df in preds:
        if not df["comp_id"].is_unique:
            raise ValueError("Identificador único 'comp_id' duplicado en predicciones.")

    output = pd.concat(preds, axis=1)
    output = output.loc[:, ~output.columns.duplicated()]
    output = pd.merge(output,data,on="comp_id")
    output = output.sort_index(axis=1)  # Ordenar columnas alfabéticamente (model_1, model_2, ...)

    tmp_dir.cleanup()
    ray.shutdown()  # Cierra Ray

    if result_path:
        output.to_csv(result_path,index=False)

    return output


