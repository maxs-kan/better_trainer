from collections import defaultdict
from fnmatch import fnmatchcase
from itertools import chain
from typing import Any, Callable

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.utilities import rank_zero_warn
from peft import LoraConfig, inject_adapter_in_model
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from ...schemas.base_module_schemas import (
    DEFAULT_GROUP,
    RESERVED_OPTION_KEYS,
    DtypeConfig,
    GradClipWarmDown,
    NetConfigs,
    NetInfo,
    OptimParamGroups,
    SchedulerConfig,
)

_WRAPPER_PREFIXES = ('_checkpoint_wrapped_module.', '_orig_mod.')
_ALLOWED_PRECISION = {
    '32-true':    {torch.float32},
    'bf16-mixed': {torch.float32, torch.bfloat16},
    '16-mixed':   {torch.float32, torch.float16}
    }
_ALLOWED_STRATEGY  = (DDPStrategy, SingleDeviceStrategy)

def _find_blocks(net: nn.Module, classes: tuple[type, ...]) -> list[nn.Module]:
    inner = {
        m._checkpoint_wrapped_module
        for m in net.modules()
        if isinstance(m, CheckpointWrapper)
        }
    return [
        m for m in net.modules()
        if m is not net and m not in inner
        and isinstance(getattr(m, "_checkpoint_wrapped_module", m), classes)
    ]

# TODO: save ema weights in f32 (broke if optimizer in bfp16), DDP mixin, adapt for inference, check config on nested activation_ckpt
class BaseModule(LightningModule):
    """
    Base class to configure trainable and freeze models, add lora
    """
    def __init__(
        self,
        net_configs: NetConfigs,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: SchedulerConfig | None = None,
        optim_param_groups: OptimParamGroups | None = None,
        grad_clip_warmdown: GradClipWarmDown | None = None,
        pop_frozen_keys_from_ckpt: bool = True,
        skip_missing_keys: bool = False,
        skip_unexpected_keys: bool = False,
    ) -> None:
        super().__init__()

        self.net_configs = net_configs
        self.optimizer_fn = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.optim_param_groups = optim_param_groups
        self.grad_clip_warmdown = grad_clip_warmdown
        self.pop_frozen_keys_from_ckpt = pop_frozen_keys_from_ckpt
        self.skip_missing_keys = skip_missing_keys
        self.skip_unexpected_keys = skip_unexpected_keys

        self._active_params = None
        self._frozen_params = None
        self._registration_closed = False
        self._cast_dtype = True
        self._nets: dict[str, NetInfo] = dict()

    def setup(self, stage: str) -> None:
        strategy = self.trainer.strategy
        if not isinstance(strategy, _ALLOWED_STRATEGY):
            raise RuntimeError(f'Unexpected strategy: {type(strategy).__name__}')
        
        precision = str(self.trainer.precision)
        if precision not in _ALLOWED_PRECISION:
            self._cast_dtype = False
            rank_zero_warn('Unexpected lightning precision, manual dtype casting is turned off')

    
    def _init_nn_modules(self):
        for net_info in self.net_configs.values():
            net = self._build_net(net_info)
            net = self.set_up_module(
                net,
                net_info.grad,
                net_info.eval_mode,
                net_info.lora_config,
                net_info.lora_adapter_name
            )
            self._cast_params_dtype(net, net_info.dtype_config)
            self._activation_checkpointing(net, net_info.activation_checkpointing)
            self._compile_net(net, net_info.compile_config)
            self._parallelize_net(net, net_info)
            self.add_nn_module(
                net_info.name,
                net,
                net_info.eval_mode,
                net_info.frozen,
            )

    def _build_net(self, net_info):
        return net_info.init_fn(**net_info.init_kwargs)

    def _parallelize_net(self, net, net_info):
        pass

    def _activation_checkpointing(self, net, config):
        if config is None:
                return
        check_fn = lambda l: isinstance(l, config.ac_modules) 
        apply_activation_checkpointing(
            net,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn)
        
    def _compile_net(self, net, config) -> None:
        if config is None or not config.compile:
            return
        if config.compile_modules:
            for block in _find_blocks(net, config.compile_modules):
                block.compile(**config.compile_kwargs())
        else:
            net.compile(**config.compile_kwargs())

    def _cast_params_dtype(self, net, config: DtypeConfig) -> None:
        if config is None or not config.cast_dtype or not self._cast_dtype :
            return
        for p in net.parameters():
            if not p.is_floating_point():
                raise RuntimeError(
                    'Unexpected non floating point dtype for parameter with manual dtype casting'
                    )
            if p.requires_grad:
                if config.active_dtype is not None:
                    p.data = p.data.to(config.active_dtype)
            else:
                if config.frozen_dtype is not None:
                    p.data = p.data.to(config.frozen_dtype)
        
        if config.buffer_dtype is not None:
            for b in net.buffers():
                if b.is_floating_point():
                    b.data = b.data.to(config.buffer_dtype)
        
        precision = str(self.trainer.precision)
        allowed = _ALLOWED_PRECISION[precision]
        all_dtypes, active_dtypes = set(), set()
        for p in net.parameters():
            if p.requires_grad:
                active_dtypes.add(p.dtype)
            all_dtypes.add(p.dtype)
        
        if not config.allow_custom_dtype_mix:
            if not all_dtypes <= allowed:
                raise RuntimeError(f'Precision {precision} allows dtype: {allowed}, got {all_dtypes}')
        
        if len(active_dtypes) > 1:
            raise RuntimeError(f'Unexpected to have multiple active dtypes: {active_dtypes}')
        
        if active_dtypes != {torch.float32}:
            if precision == '16-mixed':
                raise RuntimeError('16-mixed need fp32 trainable params for GradScaler')
            rank_zero_warn(f'Trainable fp32 master weights is recomended, got {active_dtypes}')

    def add_nn_module(
        self,
        name: str,
        net: nn.Module,
        eval_mode: bool,
        frozen: bool,
    ):
        if self.registration_closed:
            raise RuntimeError(
                f'Try to add {name} network after registration closed'
            )
        if hasattr(self, name) or name in self._nets:
            raise ValueError(f'{name} already registered')
        self._nets[name] = NetInfo(eval_mode, frozen)
        setattr(self, name, net)

    def set_up_module(
        self,
        net: nn.Module,
        grad: bool,
        eval_mode: bool,
        lora_config: LoraConfig | None = None,
        lora_adapter_name: str = "default"
    ) -> nn.Module:
        if eval_mode and grad:
            raise ValueError(f'Eval mode with enabled grad is unexpected')
        if grad and lora_config is not None:
            raise ValueError('LoRA together with full finetune (grad=true) is unexpected')
        if (not grad and lora_config is None) and not eval_mode:
            raise ValueError('Non trainable net should be in eval mode')
        if lora_config is not None and eval_mode:
            rank_zero_warn('Model with LoRA will be in eval mode, which may affects LoRA dropout')
        
        net.requires_grad_(grad)
        if lora_config is not None:
            net = self.add_lora(net, lora_config, lora_adapter_name)
        net.train(not eval_mode)       
        return net

    def add_lora(self, net: nn.Module, lora_config: LoraConfig, adapter_name="default") -> nn.Module:
        net = inject_adapter_in_model(lora_config, net, adapter_name=adapter_name)
        if not any(p.requires_grad for p in net.parameters()):
            raise RuntimeError(f"No matching for target modules: {lora_config.target_modules}")
        return net

    def _post_init_nn_modules(self) -> None:
        if not self._nets:
            raise RuntimeError('No network was registered')
        
        unregistered = [
            n for n, m in self.named_children()
            if m.state_dict() and n not in self._nets
        ]
        if unregistered:
            raise RuntimeError(f'Detected network which was not registered via add_nn_module: {unregistered}')

        direct_params = [n for n, _ in self.named_parameters(recurse=False)]
        direct_buffers = [n for n, _ in self.named_buffers(recurse=False)]
        if direct_params or direct_buffers:
            raise RuntimeError(
                f'Parameters/buffers attached directly to the module are not supported: '
                f'{direct_params + direct_buffers}. Wrap them in an nn.Module and register it via add_nn_module.'
            )
        self._collect_keys()
        if not self._active_params:
            raise RuntimeError('No active parameters to train')
        if self.pop_frozen_keys_from_ckpt:
            self.strict_loading = False
        else:
            self.strict_loading = True
        self._registration_closed = True


    def strategy_post_configure(self):
        if isinstance(self.trainer.strategy, DDPStrategy):
            self._ddp_params_and_buffers_to_ignore = [
                n for n, _ in chain(self.named_parameters(), self.named_buffers())
                if self._clean_key(n) not in self.active_params
            ]

    def configure_model(self) -> None:
        if self.registration_closed:
            return
        self._init_nn_modules()
        self._post_init_nn_modules()
        self.strategy_post_configure()

    def train(self, mode: bool = True):
        super().train(mode)
        for n, info in self._nets.items():
            if info.eval_mode:
                getattr(self, n).eval()
        return self


    @property
    def registration_closed(self):
        return self._registration_closed

    @staticmethod
    def _clean_key(k: str) -> str:
        for p in _WRAPPER_PREFIXES:
            k = k.replace(p, "")
        return k

    def _collect_keys(self) -> None:
        active, frozen = set(), set()
        for name, info in self._nets.items():
            net = getattr(self, name)
            is_trainable = not info.frozen and net.training

            for k, p in net.state_dict(keep_vars=True).items():
                key = f'{name}.{self._clean_key(k)}'
                is_active = p.requires_grad if isinstance(p, nn.Parameter) else is_trainable
                if is_active:
                    active.add(key)
                else:
                    frozen.add(key)
        self._active_params = frozenset(active)
        self._frozen_params = frozenset(frozen)


    @property
    def frozen_params(self) -> frozenset[str]:
        if not self.registration_closed:
            raise RuntimeError('Attempting module params specification before all nets registered properly')
        return self._frozen_params

    @property
    def active_params(self) -> frozenset[str]:
        if not self.registration_closed:
            raise RuntimeError('Attempting module params specification before all nets registered properly')
        return self._active_params

    def _checkpoint_validation(self, checkpoint):
        state_dict = checkpoint['state_dict']
        loaded = {self._clean_key(k) for k in state_dict.keys()}
        missing = self.active_params - loaded
        if missing and not self.skip_missing_keys:
            raise RuntimeError(f'Missing keys for active params in checkpoint: {sorted(missing)[:10]}')
        
        unexpected = loaded - self.active_params - self.frozen_params
        if unexpected and not self.skip_unexpected_keys:
            raise RuntimeError(f"Unexpected keys in checkpoint: {sorted(unexpected)[:10]}")

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.pop_frozen_keys_from_ckpt:
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if self._clean_key(k) in self.frozen_params:
                    state_dict.pop(k)

    def on_load_checkpoint(self, checkpoint) -> None:
        if not self.strict_loading:
            self._checkpoint_validation(checkpoint)


    def _set_params_groups(self, params: list[tuple[str, nn.Parameter]]):
        buckets = defaultdict(list)
        assigned = {}
        defaults = []
        for n, p in params:
            key = self._clean_key(n)
            for group in self.optim_param_groups.values():
                for g_pattern in group.pattern:
                    if fnmatchcase(key, g_pattern):
                        if assigned.get(n):
                            raise RuntimeError(f'Parameter {n} is assigned to group {assigned[n]} and {group.name}')
                        buckets[group.name].append((n, p))
                        assigned[n] = group.name
                        break
            if assigned.get(n) is None:
                defaults.append((n, p))
                assigned[n] = DEFAULT_GROUP
        groups = []
        if defaults:
            groups.append({'params': defaults, 'name': DEFAULT_GROUP})
        for group in self.optim_param_groups.values():
            g_params = buckets[group.name]
            if not g_params:
                raise RuntimeError(f'No matched parameters for group {group.name}: {group.pattern}')
            opt = {'params': g_params, 'name': group.name, **group.options}
            if group.lr is not None:
                opt['lr'] = group.lr
            if group.weight_decay is not None:
                opt['weight_decay'] = group.weight_decay
            groups.append(opt)
        return groups

    def _create_optim_groups(self):
        params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if not self._clean_key(n) in self.active_params:
                    raise RuntimeError(f'Parameter: {n} is required grad, but not in configured active parameters')
                params.append((n, p))
        if not self.optim_param_groups:
            return [{'params': params}]
        return self._set_params_groups(params)

    def _check_group_options(self, optimizer: torch.optim.Optimizer) -> None:
        allowed = set(optimizer.defaults) | RESERVED_OPTION_KEYS
        for group in optimizer.param_groups:
            unknown = set(group) - allowed
            if unknown:
                raise ValueError(
                    f"Param group {group.get('name')} got options not supported by "
                    f"{type(optimizer).__name__}: {sorted(unknown)}"
                )

    def configure_optimizers(self) -> dict[str, Any]:
        params = self._create_optim_groups()
        optimizer = self.optimizer_fn(params=params)
        self._check_group_options(optimizer)
        cfg = self.lr_scheduler_config
        if cfg is not None:
            scheduler = cfg.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": cfg.monitor,
                    "interval": cfg.interval,
                    "frequency": cfg.frequency,
                },
            }
        return {"optimizer": optimizer}

    def _grad_norm_val(self) -> float | None:
        cfg = self.grad_clip_warmdown
        step, total = self.global_step, cfg.warmdown_steps
        if total <= 0 or step >= total:
            return cfg.base_grad_norm
        return cfg.max_grad_norm + step * (cfg.base_grad_norm - cfg.max_grad_norm) / total

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None) -> None:
        if self.grad_clip_warmdown is not None:
            gradient_clip_val = self._grad_norm_val()
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )
