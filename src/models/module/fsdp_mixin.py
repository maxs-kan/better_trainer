from lightning.pytorch.utilities import rank_zero_warn
from torch.distributed.fsdp import fully_shard
import torch
import torch.nn as nn

from ...schemas.fsdp_schemas import FSDPConfig, FSDPNetConfigs, FSDPNetConfig
from ...strategy.model_parallel_base import ParallelStrategyBase
from .base_module import _find_blocks


class FSDPMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.net_configs, FSDPNetConfigs):
            raise ValueError(f"Configure models with FSDPNetConfigs, got {type(self.net_configs).__name__}")
        if self.pop_frozen_keys_from_ckpt:
            rank_zero_warn('No checkpoint validation available with partial checkpoint under FSDP')

        self._drop_frozen_from_state = False

    def setup(self, stage: str) -> None:
        if not isinstance(self.trainer.strategy, ParallelStrategyBase):
            raise RuntimeError(f'{type(self).__name__} requires src.strategy.model_parallel_base.ParallelStrategyBase')

    def _build_net(self, net_info):
        with torch.device("meta"):
            return super()._build_net(net_info)

    def set_up_module(self, *args, **kwargs):
        with torch.device("meta"):
            net = super().set_up_module(*args, **kwargs)
        return net

    def on_fit_start(self) -> None:
        raise NotImplementedError('You should implement model weight loading into device')

    def _parallelize_net(self, net: nn.Module, net_info: FSDPNetConfig):
        self._shard_net(net, net_info.fsdp)

    def _shard_net(self, net: nn.Module, config: FSDPConfig):
        dp_mesh = self.device_mesh["data_parallel"]
        if config.shard_modules:
            for block in reversed(_find_blocks(net, config.shard_modules)):
                fully_shard(block, mesh=dp_mesh, **config.fully_shard_kwargs())
        fully_shard(net, mesh=dp_mesh, **config.fully_shard_kwargs())
    
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self._drop_frozen_from_state:
            for k in list(state_dict.keys()):
                if self._clean_key(k) in self.frozen_params:
                    state_dict.pop(k)
        return state_dict

    def on_load_checkpoint(self, checkpoint) -> None:
        pass
    
    #TODO: add cpu offload, prefetching
    #TODO: call for BaseModule setup
    #TODO: add weight loadint directly to device see https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/tensor_parallel/train.py
    #TODO: weights not broacasted accros device, may be crucial for random weights
    #TODO: check how lightning mixed precision should be set for fsdp2
    #TODO: EMA loading from checkpoint
