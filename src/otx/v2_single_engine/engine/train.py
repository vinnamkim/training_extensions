# import hydra

# import rootutils

# from omegaconf import DictConfig

# from otx.v2_single_engine.engine import utils

# rootutils.setup_root(__file__, indicator=".engine-root", pythonpath=True)
# # ------------------------------------------------------------------------------------ #
# # the setup_root above is equivalent to:
# # - adding project root dir to PYTHONPATH
# #       (so you don't need to force user to install project as a package)
# #       (necessary before importing any local modules e.g. `from src import utils`)
# # - setting up PROJECT_ROOT environment variable
# #       (which is used as a base for paths in "configs/paths/default.yaml")
# #       (this way all filepaths are the same no matter where you run the code)
# # - loading environment variables from ".env" in root dir
# #
# # you can remove it if you:
# # 1. either install project as a package or move entry files to project root dir
# # 2. set `root_dir` to "." in "configs/paths/default.yaml"
# #
# # more info: https://github.com/ashleve/rootutils
# # ------------------------------------------------------------------------------------ #


import logging as log
from typing import Any, Dict, List, Tuple

import hydra
from omegaconf import DictConfig

from otx.v2_single_engine.config_structs.train import TrainConfig, register_configs

# log = utils.get_pylogger(__name__)


# @utils.task_wrapper
def train(cfg: TrainConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import lightning as L
    from lightning import Callback, LightningModule, Trainer
    from lightning.pytorch.loggers import Logger

    from otx.v2_single_engine.data_module import OTXDataModule
    from otx.v2_single_engine.engine import utils

    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data}>")
    datamodule = OTXDataModule(task=cfg.base.task, config=cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


register_configs()


@hydra.main(version_base="1.3", config_path="../base_configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = utils.get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value
    print(type(cfg.task))


if __name__ == "__main__":
    main()