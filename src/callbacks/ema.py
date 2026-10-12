import torch
import lightning.pytorch as pl
from lightning.pytorch import Callback


EMA_KEY = 'ema'


def get_params(optimizer: torch.optim.Optimizer):
    return [p for g in optimizer.param_groups for p in g['params']]

class EMA(Callback):
    def __init__(
        self,
        decay: float,
        start_step: int = 0,
        every_n_step: int = 1,
        validation_with_ema: bool = False
    ):
        self.decay = decay
        self.start_step = start_step
        self.every_n_step = every_n_step
        self.validation_with_ema = validation_with_ema

        self._handlers = []
        self._optimizers: list[torch.optim.Optimizer] = []
        self._weights_swapped = False
        self._weight_backup: dict[torch.nn.Parameter, torch.Tensor] | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._optimizers = list(trainer.optimizers)
        for opt in self._optimizers:
            self._handlers.append(opt.register_step_post_hook(lambda o, *_ : self.update(o, trainer.global_step)))
        return super().on_fit_start(trainer, pl_module)

    @torch.no_grad()
    def update(self, optimizer: torch.optim.Optimizer, global_step: int):
        if global_step < self.start_step or global_step % self.every_n_step:
            return

        params = [p for p in get_params(optimizer) if p.grad is not None]
        for p in params:
            state = optimizer.state[p]
            if EMA_KEY not in state:
                state[EMA_KEY] = p.detach().float().clone()

        emas = [optimizer.state[p][EMA_KEY] for p in params]
        src = [p.detach().float() for p in params]
        weight = 1.0 - self.decay
        torch._foreach_lerp_(emas, src, weight)

    @torch.no_grad()
    def _load_ema(self):
        self._weight_backup = {}
        for optimizer in self._optimizers:
            params = get_params(optimizer)
            for p in params:
                e = optimizer.state[p].get(EMA_KEY)
                if e is None:
                    continue
                self._weight_backup[p] = p.detach().clone()
                p.detach().copy_(e)
        self._weights_swapped = True

    @torch.no_grad()
    def _unload_ema(self):
        for p, w in self._weight_backup.items():
            p.detach().copy_(w)
        self._weight_backup = None
        self._weights_swapped = False

    
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.validation_with_ema and not self._weights_swapped:
            self._load_ema()
        return super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._weights_swapped:
            self._unload_ema()
        return super().on_validation_end(trainer, pl_module)

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "fit" and self.validation_with_ema:
            raise RuntimeError(
                f"validation_with_ema is set but stage={stage} has no optimizers. "
                "Load EMA weights into the model with ema_weights_from_checkpoint() instead."
            )
        return super().setup(trainer, pl_module, stage)

    def finalize(self):
        if self._weights_swapped:
            self._unload_ema()
        for h in self._handlers:
            h.remove()
        self._handlers.clear()
        self._optimizers.clear()
        self._weight_backup = None


    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.finalize()
        return super().on_fit_end(trainer, pl_module)


    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        self.finalize()
        return super().on_exception(trainer, pl_module, exception)


def ema_weights_from_checkpoint(
    ckpt: dict, optimizer_idx: int = 0, clean_key=None
) -> dict[str, torch.Tensor]:

    opt_state = ckpt["optimizer_states"][optimizer_idx]
    state = opt_state["state"]
    clean = clean_key or (lambda k: k)

    # FSDP full optimizer state dict: state is already keyed by parameter name
    if state and all(isinstance(k, str) for k in state):
        weights = {clean(k): v[EMA_KEY] for k, v in state.items() if EMA_KEY in v}
    else:
        groups = opt_state["param_groups"]
        if any("param_names" not in g for g in groups):
            raise RuntimeError(
                "Checkpoint has no param_names; build the optimizer from "
                "named parameters: optimizer_fn(params=[(n, p), ...])"
            )
        weights = {
            clean(name): state[i][EMA_KEY]
            for g in groups
            for i, name in zip(g["params"], g["param_names"])
            if i in state and EMA_KEY in state[i]
        }

    if not weights:
        raise RuntimeError("No EMA state in checkpoint (saved before start_step?)")
    return weights
