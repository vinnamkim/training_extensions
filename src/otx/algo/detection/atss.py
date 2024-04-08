# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""ATSS model implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from otx.algo.utils.mmconfig import read_mmconfig
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.mmdeploy import MMdeployExporter
from otx.core.metrics.mean_ap import MeanAPCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.detection import MMDetCompatibleModel
from otx.core.schedulers import LRSchedulerListCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class ATSS(MMDetCompatibleModel):
    """ATSS Model."""

    def __init__(
        self,
        num_classes: int,
        variant: Literal["mobilenetv2", "resnext101"],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAPCallable,
        torch_compile: bool = False,
    ) -> None:
        model_name = f"atss_{variant}"
        config = read_mmconfig(model_name=model_name)
        super().__init__(
            num_classes=num_classes,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_size = (1, 3, 736, 992)
        self.tile_image_size = self.image_size

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.image_size is None:
            raise ValueError(self.image_size)

        return MMdeployExporter(
            model_builder=self._create_model,
            model_cfg=deepcopy(self.config),
            deploy_cfg="otx.algo.detection.mmdeploy.atss",
            test_pipeline=self._make_fake_test_pipeline(),
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            output_names=["feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_det_ckpt(state_dict, add_prefix)
