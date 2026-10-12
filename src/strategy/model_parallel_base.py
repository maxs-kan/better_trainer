from contextlib import contextmanager
from typing import Any
from lightning.pytorch.strategies import ModelParallelStrategy
from pathlib import Path
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from ..callbacks.ema import EMA, EMA_KEY


def _add_ema_placeholders(self, path: Path) -> None:
    module = self.lightning_module
    if not any(isinstance(cb, EMA) for cb in module.trainer.callbacks):
        return                                     
    if not path.is_dir():
        return
    saved = FileSystemReader(path).read_metadata().state_dict_metadata
    names = {id(p): module._clean_key(n) for n, p in module.named_parameters()}
    for idx, opt in enumerate(self.optimizers):
        get_optimizer_state_dict(self.model, opt)
        for g in opt.param_groups:
            for p in g["params"]:
                if f"optimizer_{idx}.state.{names[id(p)]}.{EMA_KEY}" in saved:
                    opt.state[p].setdefault(EMA_KEY, p.detach().float().clone())

class ParallelStrategyBase(ModelParallelStrategy):
    @contextmanager
    def _drop_frozen(self):
        module = self.lightning_module
        enabled = getattr(module, 'pop_frozen_keys_from_ckpt', False)
        module._drop_frozen_from_state = enabled
        try:
            yield
        finally:
            module._drop_frozen_from_state = False


    def load_checkpoint(self, *args, **kwargs) -> dict[str, Any]:
        with self._drop_frozen():
            return super().load_checkpoint(*args, **kwargs)