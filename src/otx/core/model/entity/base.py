# We have two nice abstracts: 1) Task and 2) Data entity (Datumaro annotations)
# Since the task determines the input and output format of the model, we can define the following abstract model class for each task.
# At this time, what OTX model developers need to do is very clear: Implement the two abstract functions.

from abc import abstractmethod
from typing import Any, Dict

from torch import nn

from otx.core.data.entity.base import (
    OTXBatchDataEntity,
    OTXBatchLossEntity,
    OTXBatchPredEntity,
)


class OTXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.create_model()

    @abstractmethod
    def create_model(self) -> nn.Module:
        pass

    def customize_inputs(self, inputs: OTXBatchDataEntity) -> Dict[str, Any]:
        raise NotImplementedError

    def customize_outputs(self, outputs: Any) -> OTXBatchPredEntity:
        raise NotImplementedError

    def forward(
        self,
        inputs: OTXBatchDataEntity,
    ) -> OTXBatchLossEntity | OTXBatchPredEntity:
        # If customize_inputs is overrided
        outputs = (
            self.model(**self.customize_inputs(inputs))
            if self.customize_inputs != OTXModel.customize_inputs
            else self.model(inputs)
        )

        return (
            self.customize_outputs(outputs)
            if self.customize_outputs != OTXModel.customize_outputs
            else outputs
        )
