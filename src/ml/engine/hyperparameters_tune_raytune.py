from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune import CLIReporter
from lightning import Trainer
from lightning.pytorch import seed_everything
import pandas as pd
import os


from utils.configurations import configs
from utils.constants import RANDOM_STATE
from datasets.datamodule import DEEPscreenDataModule
from engine.system import DEEPScreenClassifier


class deepscreen_hyperparameter_tuneing:
    def __init__(self,data:pd.DataFrame,search_space:dict,target:str,max_epochs:int = 200,data_split_mode:str="non_random_split", grace_period:int=35,metric:str="val_mcc",mode:str="max",num_workers:int=1,num_samples:int=10,experiments_result_path="../../.experiments/"):
        seed_everything(RANDOM_STATE,True)

        self.data = data
        self.search_space = search_space
        self.num_samples =  num_samples
        self.target = target
        self.data_split_mode = data_split_mode
        self.experiment_path_abs = os.path.abspath(experiments_result_path)
        self.experiment_result_path = os.path.join( self.experiment_path_abs,self.target)
        self.max_epochs = max_epochs
        self.grace_period = grace_period

        if not os.path.exists(self.experiment_result_path):
                os.makedirs(self.experiment_result_path)

        self.scheduler = ASHAScheduler(
            time_attr="epoch",
            metric="val_mcc",
            mode="max",
            max_t=self.max_epochs,
            grace_period=self.grace_period,
            reduction_factor=3,
            )

        self.scaling_config = ScalingConfig(**configs.get_raytune_scaleing_config())

        self.run_config = RunConfig(
            stop={"training_iteration": self.max_epochs},
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode
            )
            )

        self.ray_trainer = TorchTrainer(
            self._train_func,
            scaling_config=self.scaling_config,
            run_config=self.run_config,
        )

    def _train_func(self,config):
        dm = DEEPscreenDataModule(
             data=self.data,
             target_id=self.target,
             batch_size=config["batch_size"],
             experiment_result_path=self.experiment_result_path,
             data_split_mode=self.data_split_mode,
             tmp_imgs=configs.get_use_tmp_imgs())
        
        model = DEEPScreenClassifier(**config,experiment_result_path=self.experiment_result_path)

        number_training_batches = round(len(self.data)*0.64/config["batch_size"])

        trainer = Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            deterministic=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            max_epochs = self.max_epochs,
            log_every_n_steps=number_training_batches
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    def tune_deepscreen(self):

        tuner = tune.Tuner(
            self.ray_trainer,
            param_space={"train_loop_config": self.search_space},
            tune_config=tune.TuneConfig(         
                num_samples=self.num_samples,
                scheduler=self.scheduler,
            ),
        )
        return tuner.fit()
