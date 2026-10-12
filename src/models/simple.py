from typing import Callable

import torch
import torch.nn as nn

from .module.base_module import BaseModule
from .module.fsdp_mixin import FSDPMixin


class SimpleModule(BaseModule):
    def __init__(
        self,
        criterion,
        **kwargs
    ):
        super().__init__(**kwargs)
        criterion = self.set_up_module(criterion, False, True)
        self.add_nn_module('criterion', criterion, True, True)

    def forward(self, batch: dict, keep_batch=False) -> torch.Tensor:
        y = self.ffn(batch['x'])
        return y

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        loss = self.criterion(preds, batch['y'])
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=batch['y'].shape[0], sync_dist=True)
        return loss


class SimpleModuleFSDP(FSDPMixin, SimpleModule):
    pass