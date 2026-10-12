from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn as nn
from hydra.utils import get_object
from omegaconf import MISSING, OmegaConf
from peft import LoraConfig

from ..utils.instantiators import instantiate_dtype

COMPILE_MODES = frozenset({
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
})
SCHEDULER_INTERVALS = frozenset({"step", "epoch"})
DEFAULT_GROUP = "default"
RESERVED_OPTION_KEYS = frozenset({"params", "param_names", "name", "lr", "weight_decay"})
TORCH_FLOAT_DTYPES = frozenset({'float32', 'bfloat16', 'float16'})


def _float_dtype(field_name: str, value: Any) -> torch.dtype | None:
    if value is None:
        return None
    name = str(value).removeprefix("torch.")
    if name not in TORCH_FLOAT_DTYPES:
        raise ValueError(f"{field_name} must be one of {sorted(TORCH_FLOAT_DTYPES)}, got {value}")
    return instantiate_dtype(name)

def _todataclass(value, cls, config_name: str, **kwargs):
    if value is None:
        return None
    if isinstance(value, cls):
        return value
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return cls(**value, **kwargs)
    raise TypeError(f"{config_name} must be a mapping or {cls.__name__}, got {type(value).__name__}")


@dataclass
class NetInfo:
    eval_mode: bool
    frozen: bool

@dataclass
class SchedulerConfig:
    _target_: str = f"{__name__}.SchedulerConfig"
    scheduler: Any = MISSING
    interval: str = "step"
    frequency: int = 1
    monitor: Optional[str] = None

    def __post_init__(self) -> None:
        if self.interval not in SCHEDULER_INTERVALS:
            raise ValueError(
                f"interval must be one of {sorted(SCHEDULER_INTERVALS)}, got {self.interval}"
            )
        if self.frequency < 1:
            raise ValueError(f"frequency must be >= 1, got {self.frequency}")


@dataclass
class GradClipWarmDown:
    warmdown_steps: int
    base_grad_norm: float
    max_grad_norm: float
    _target_: str = f"{__name__}.GradClipWarmDown"

    def __post_init__(self) -> None:
        if self.warmdown_steps < 0:
            raise ValueError(f"warmdown_steps must be an int >= 0, got {self.warmdown_steps}")
        if self.base_grad_norm <= 0:
            raise ValueError(f"base_grad_norm must be > 0, got {self.base_grad_norm}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        if self.max_grad_norm < self.base_grad_norm:
            raise ValueError(
                f"max_grad_norm ({self.max_grad_norm}) must be >= base_grad_norm ({self.base_grad_norm}): "
                "the clip limit warms down from max_grad_norm to base_grad_norm"
            )


@dataclass
class OptimParamGroup:
    name: str
    pattern: Any
    lr: Optional[float] = None
    weight_decay: Optional[float]  = None
    options: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.pattern, str):
            self.pattern = [self.pattern]
        self.pattern = list(self.pattern)
        if not self.pattern or not all(isinstance(p, str) and p for p in self.pattern):
            raise ValueError(f"Group {self.name} needs non-empty string patterns, got {self.pattern}")
        if self.lr is not None and self.lr <= 0:
            raise ValueError(f"Group {self.name}: lr must be > 0, got {self.lr}")
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError(f"Group {self.name}: weight_decay must be >= 0, got {self.weight_decay}")
        reserved = RESERVED_OPTION_KEYS & set(self.options)
        if reserved:
            raise ValueError(f"Group {self.name}: options may not set {reserved}")
        self.options = {k: tuple(v) if isinstance(v, list) else v for k, v in self.options.items()}

@dataclass
class OptimParamGroups(Mapping):
    _target_: str = f"{__name__}.OptimParamGroups"
    groups: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.groups:
            if name == DEFAULT_GROUP:
                raise ValueError(f"Group name {DEFAULT_GROUP} is reserved for unmatched params")
        self.groups = {
            name: _todataclass(cfg, OptimParamGroup, f'optim_group_config', name=name)
            for name, cfg in self.groups.items()
            if cfg is not None
        }

    def __iter__(self):
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, name: str) -> OptimParamGroup:
        return self.groups[name]

    def __repr__(self) -> str:
        return f"OptimParamGroups({list(self.groups)})"


def _resolve_init_fn(net_name: str, value: Any) -> Callable[..., nn.Module]:
    if isinstance(value, nn.Module):
        raise TypeError(
            f"{net_name}.init_fn was instantiated into a module at config time; "
            f"add `_partial_: true` or pass a plain import path string"
        )
    if OmegaConf.is_config(value) or isinstance(value, dict):
        raise TypeError(f"{net_name}.init_fn must be an import path " 
                        f"or a `_target_` + `_partial_: true + _recursive_: true` node")
    if isinstance(value, str):
        try:
            value = get_object(value)
        except Exception as e:
            raise ValueError(f"{net_name}.init_fn: cannot import {value}") from e
    if not callable(value):
        raise TypeError(f"{net_name}.init_fn must be callable, got {type(value).__name__}")
    return value

def _resolve_nn_classes(field_name: str, value: Any) -> tuple[type[nn.Module], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, type)):
        value = [value]
    resolved = []
    for m in value:
        if isinstance(m, str) and m:
            try:
                m = get_object(m)
            except Exception as e:
                raise ValueError(f"{field_name}: cannot import {m}") from e
        if not (isinstance(m, type) and issubclass(m, nn.Module)):
            raise TypeError(f"{field_name}: {m} is not an nn.Module class")
        resolved.append(m)
    return tuple(resolved)

@dataclass
class ActivationCheckpointingConfig:
    ac_modules: Any = None

    def __post_init__(self) -> None:
        self.ac_modules = _resolve_nn_classes("activation_checkpointing_modules", self.ac_modules)
        if not self.ac_modules:
            raise ValueError("activation_checkpointing_modules must has at least one module class")

@dataclass
class CompileConfig:
    compile: bool = False
    dynamic: bool = False
    mode: str = "default"
    fullgraph: bool = False
    compile_modules: Any = None

    def __post_init__(self) -> None:
        if self.mode not in COMPILE_MODES:
            raise ValueError(
                f"Unknown compile mode {self.mode}, expected one of {sorted(COMPILE_MODES)}"
            )
        self.compile_modules = _resolve_nn_classes("compile_modules", self.compile_modules)

    def compile_kwargs(self):
        kw = {
            'dynamic': self.dynamic,
            'mode': self.mode,
            'fullgraph': self.fullgraph
        }
        return kw

@dataclass
class DtypeConfig:
    _target_: str = f"{__name__}.DtypeConfig"
    cast_dtype: bool = False
    active_dtype: Any = "float32"
    frozen_dtype: Any = "float32"
    buffer_dtype: Any = None
    allow_custom_dtype_mix: bool = False

    def __post_init__(self) -> None:
        self.active_dtype = _float_dtype("active_dtype", self.active_dtype)
        self.frozen_dtype = _float_dtype("frozen_dtype", self.frozen_dtype)
        self.buffer_dtype = _float_dtype("buffer_dtype", self.buffer_dtype)

@dataclass
class NetConfig:
    name: str
    init_fn: Any
    eval_mode: bool
    grad: bool
    lora_config: Any = None
    lora_adapter_name: str = "default"
    activation_checkpointing: Any = None
    compile_config: Any = None
    dtype_config: Any = None
    init_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Net name {self.name} must be a valid python identifier")
        if self.init_kwargs is None:
            self.init_kwargs = {}
        if not isinstance(self.init_kwargs, dict):
            raise TypeError(f"init_kwargs must be a mapping, got {type(self.init_kwargs).__name__}")
        if self.eval_mode and self.grad:
            raise ValueError(f"{self.name}: eval mode with enabled grad is unexpected")
        if self.grad and self.lora_config is not None:
            raise ValueError(f"{self.name}: LoRA together with full finetune (grad=true) is unexpected")
        if self.frozen and not self.eval_mode:
            raise ValueError(f"{self.name}: non trainable net should be in eval mode")
        
        self.init_fn = _resolve_init_fn(self.name, self.init_fn)
        self.lora_config = _todataclass(self.lora_config, LoraConfig, f'{self.name}.lora_config')
        self.activation_checkpointing = _todataclass(
            self.activation_checkpointing,
            ActivationCheckpointingConfig,
            f"{self.name}.activation_checkpointing_config"
            )
        self.compile_config = _todataclass(
            self.compile_config,
            CompileConfig,
            f"{self.name}.compile_config"
        )
        self.dtype_config = _todataclass(
            self.dtype_config,
            DtypeConfig,
            f"{self.name}.dtype_config"
        )
        if self.frozen and self.activation_checkpointing is not None:
            raise ValueError(f"{self.name}: activation checkpointing on a frozen net is useless")

    @property
    def frozen(self) -> bool:
        return not self.grad and self.lora_config is None

class NetConfigs(Mapping[str, NetConfig]):
    net_config_cls: type[NetConfig] = NetConfig
    
    def __init__(self, **nets: Any) -> None:
        self._nets: dict[str, NetConfig] = {}
        for name, cfg in nets.items():
            if cfg is None:
                continue
            info = _todataclass(cfg, self.net_config_cls, f'{name}.net_config', name=name)
            self._nets[name] = info
        if not self._nets:
            raise ValueError("nets_config has no nets")

    def __getitem__(self, name: str) -> NetConfig:
        return self._nets[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._nets)

    def __len__(self) -> int:
        return len(self._nets)

    def __repr__(self) -> str:
        return f"NetConfigs({list(self._nets)})"
