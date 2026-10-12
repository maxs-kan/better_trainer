from dataclasses import dataclass


@dataclass
class EmaConfig:
    _target_: str = "schemas.callbacks.EmaConfig"
    decay: float = 0.9999
    start_step: int = 0
    eval_with_ema: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")