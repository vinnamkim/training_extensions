import pytest
from otx.v2_single_engine.data_entity.detection import DetBatchDataEntity
from otx.v2_single_engine.data_module import OTXDataModule
from otx.v2_single_engine.types.task import OTXTaskType
from otx.v2_single_engine.config_structs.data_module import (
    DataModuleConfig,
    SubsetConfig,
)
from otx.v2_single_engine.model.detection.mmdet import MMDetCompatibleModel
import os.path as osp
from otx.v2_single_engine.utils.config import mmconfig_dict_to_dict
from omegaconf import DictConfig


class TestOTXModel:
    @pytest.fixture
    def fxt_rtmdet_tiny_model_config(self, fxt_rtmdet_tiny_config):
        return DictConfig(mmconfig_dict_to_dict(fxt_rtmdet_tiny_config.model))

    @pytest.fixture
    def fxt_model(self, fxt_rtmdet_tiny_model_config) -> MMDetCompatibleModel:
        return MMDetCompatibleModel(config=fxt_rtmdet_tiny_model_config)

    def test_forward_train(
        self, fxt_model: MMDetCompatibleModel, fxt_datamodule: OTXDataModule
    ):
        dataloader = fxt_datamodule.train_dataloader()
        for inputs in dataloader:
            outputs = fxt_model(inputs)
            assert isinstance(outputs, dict)
            break
