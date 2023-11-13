from otx.v2_single_engine.data_entity.detection import DetBatchEntity
from otx.v2_single_engine.data_module import OTXDataModule


class TestOTXDataModule:
    def test_train_dataloader(self, fxt_datamodule: OTXDataModule):
        for batch in fxt_datamodule.train_dataloader():
            assert isinstance(batch, DetBatchEntity)
