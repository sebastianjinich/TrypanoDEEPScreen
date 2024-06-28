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
from datasets.datamodule_rdkit_features import DEEPscreenDataModule
from engine.system_rdkit_features import DEEPScreenClassifier


class deepscreen_hyperparameter_tuneing:
    def __init__(self,data:pd.DataFrame,search_space:dict,target:str,max_epochs:int,data_split_mode:str,grace_period:int,metric_to_optimize:str,optimize_mode:str,num_samples:int,experiments_result_path:str,asha_reduction_factor:int,number_ckpts_keep:int):
        seed_everything(RANDOM_STATE,True)

        self.data = data
        self.search_space = search_space
        self.num_samples =  num_samples
        self.target = target
        self.data_split_mode = data_split_mode
        self.experiment_path_abs = os.path.abspath(experiments_result_path)
        self.experiment_result_path = self.experiment_path_abs
        self.max_epochs = max_epochs
        self.grace_period = grace_period
        self.metric = metric_to_optimize
        self.mode = optimize_mode
        self.reduction_factor = asha_reduction_factor

        os.environ["TUNE_RESULT_DIR"] = self.experiment_path_abs
        os.environ["RAY_AIR_NEW_OUTPUT"]="0"

        if not os.path.exists(self.experiment_result_path):
                os.makedirs(self.experiment_result_path)

        self.scheduler = ASHAScheduler(
            time_attr="epoch",
            metric=self.metric,
            mode=self.mode,
            max_t=self.max_epochs,
            grace_period=self.grace_period,
            reduction_factor=self.reduction_factor,
            )

        self.scaling_config = ScalingConfig(**configs.get_raytune_scaleing_config())

        self.reporter = CLIReporter(metric=self.metric,mode=self.mode,max_progress_rows=15,metric_columns=["train_loss","val_loss","val_mcc","val_f1","val_auroc","val_auroc_15"])

        self.run_config = RunConfig(
            stop={"training_iteration": self.max_epochs},
            checkpoint_config=CheckpointConfig(
                num_to_keep=number_ckpts_keep,
                checkpoint_score_attribute=self.metric,
                checkpoint_score_order=self.mode
            ),
            name=self.target,
            progress_reporter=self.reporter
            )
        

        self.ray_trainer = TorchTrainer(
            self._train_func,
            scaling_config=self.scaling_config
        )

    def _train_func(self,config):
        dm = DEEPscreenDataModule(
             data=self.data,
             batch_size=config["batch_size"],
             experiment_result_path=self.experiment_result_path,
             data_split_mode=self.data_split_mode,
             tmp_imgs=configs.get_use_tmp_imgs())
        
        model = DEEPScreenClassifier(**config,experiment_result_path=self.experiment_result_path,target=self.target)

        number_training_batches = dm.get_number_training_batches()

        trainer = Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
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
                scheduler=self.scheduler
            ),
            run_config=self.run_config
        )
        return tuner.fit()
