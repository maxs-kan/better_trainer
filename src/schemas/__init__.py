from hydra.core.config_store import ConfigStore

from .base_module_schemas import (
    CompileConfig,
    DtypeConfig,
    OptimParamGroups,
    SchedulerConfig,
    GradClipWarmDown
)

__all__ = [
    "CompileConfig",
    "SchedulerConfig",
    "DtypeConfig",
    "OptimParamGroups",
    "register_configs"]


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model/lr_scheduler", name="scheduler_schema", node=SchedulerConfig)
    cs.store(group="model/optim_param_groups", name="optim_param_groups_schema", node=OptimParamGroups)
    cs.store(group="model/grad_clip_warmdown", name="grad_clip_schema", node=GradClipWarmDown)