from dataclasses import dataclass
from typing import Any, Optional, Union

from torch.distributed.fsdp import MixedPrecisionPolicy

from .base_module_schemas import (
    NetConfig,
    NetConfigs,
    _float_dtype,
    _todataclass, 
    _resolve_nn_classes
)


@dataclass
class MPConfig:
    param_dtype: Any = None
    reduce_dtype: Any = None
    output_dtype: Any = None
    cast_forward_inputs: bool = True

    def __post_init__(self) -> None:
        self.param_dtype = _float_dtype("param_dtype", self.param_dtype)
        self.reduce_dtype = _float_dtype("reduce_dtype", self.reduce_dtype)
        self.output_dtype = _float_dtype("output_dtype", self.output_dtype)

    def to_policy(self) -> MixedPrecisionPolicy:
        return MixedPrecisionPolicy(
            param_dtype=self.param_dtype,
            reduce_dtype=self.reduce_dtype,
            output_dtype=self.output_dtype,
            cast_forward_inputs=self.cast_forward_inputs,
        )


@dataclass
class FSDPConfig:
    mp_policy: Any 
    reshard_after_forward: Optional[Union[bool, int]] = None                               
    shard_modules: Any = None

    def __post_init__(self) -> None:
        r = self.reshard_after_forward
        if r is not None and not isinstance(r, bool) and not (isinstance(r, int) and r > 0):
            raise ValueError(f"reshard_after_forward must be bool, positive int or None, got {r}")

        if self.mp_policy is None:
            raise ValueError("fsdp.mp_policy is required")        
        if isinstance(self.mp_policy, MPConfig):
            self.mp_policy = self.mp_policy.to_policy()
        elif not isinstance(self.mp_policy, MixedPrecisionPolicy):
            self.mp_policy = _todataclass(self.mp_policy, MPConfig, "fsdp.mp_policy").to_policy()

        self.shard_modules = _resolve_nn_classes('shard_modules', self.shard_modules)

    def fully_shard_kwargs(self) -> dict[str, Any]:
        kw = {"mp_policy": self.mp_policy}
        if self.reshard_after_forward is not None:
            kw["reshard_after_forward"] = self.reshard_after_forward
        return kw


@dataclass
class FSDPNetConfig(NetConfig):
    fsdp: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fsdp = _todataclass(self.fsdp, FSDPConfig, f"{self.name}.fsdp")
        if self.fsdp is None:
            raise ValueError(f"{self.name}: fsdp config is required for FSDP nets")


class FSDPNetConfigs(NetConfigs):
    net_config_cls = FSDPNetConfig