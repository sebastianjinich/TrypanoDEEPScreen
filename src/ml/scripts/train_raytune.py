from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
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
    def __init__(self,data:pd.DataFrame,search_space:dict,target:str,max_epochs:int = 200,data_split_mode:str="non_random_split",scheduler_type = "BOHB", grace_period:int=90,metric:str="val_mcc",mode:str="max",num_workers:int=1,num_samples:int=10,experiments_result_path="../../.experiments/"):
        seed_everything(RANDOM_STATE,True)

        self.data = data
        self.search_space = search_space
        self.num_samples =  num_samples
        self.target = target
        self.data_split_mode = data_split_mode
        self.experiment_path = experiments_result_path
        self.experiment_result_path = os.path.join(experiments_result_path,self.target)
        self.max_epochs = max_epochs
        if not os.path.exists(self.experiment_result_path):
                os.makedirs(self.experiment_result_path)
        self.use_gpu = configs.get_gpu_number() > 0

        self.scheduler = HyperBandForBOHB(
                            time_attr="epoch",
                            max_t=self.max_epochs,
                            reduction_factor=3,
                            stop_last_trials=True,
                        )

        self.scaling_config = ScalingConfig(
            num_workers=1, use_gpu=self.use_gpu, resources_per_worker={"CPU": configs.get_cpu_number(), "GPU": configs.get_gpu_number()}
            )

        self.search_algorithm = TuneBOHB()

        self.run_config = RunConfig(
            stop={"training_iteration": self.max_epochs},
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,

            ),
            progress_reporter=CLIReporter(max_column_length=10,max_report_frequency=15)
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

        trainer = Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            deterministic=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            max_epochs = self.max_epochs
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    def tune_deepscreen(self):

        tuner = tune.Tuner(
            self.ray_trainer,
            param_space={"train_loop_config": self.search_space},
            tune_config=tune.TuneConfig(
                metric="val_mcc",
                mode="max",
                num_samples=self.num_samples,
                scheduler=self.scheduler,
                search_alg=self.search_algorithm
            ),
        )
        return tuner.fit()
