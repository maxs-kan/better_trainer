import torch
from einops import einsum

from .base_module import BaseModule


class PDDTrajectory:
    def __init__(
        self,
        time_grid,
        L_min,
        shape,
        device,
        dtype=torch.float32,
    ):
        
        self.time_grid = time_grid.to(device=device, dtype=dtype)
        self.time_diff = time_grid.diff().to(device=device, dtype=dtype)
        if len(self.time_diff) % L_min != 0:
            raise ValueError(f"Can not divide trajectory with num steps {len(self.time_diff)} into integer number of blocks with size {L_min}")
        self.trajectory_steps = len(self.time_diff)
        self.L_min = L_min
        self.shape = shape
        self.device = device
        self.dtype = dtype

        self._state_count = None
        self._state = None


    def reset(self, generator):
        self._state = torch.randn(
            *self.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        self._state_count = 0

    @torch.no_grad()
    def update(self, pred):
        r"""Advance the state by `L_min` Euler steps with student velocities `pred` of shape `(b, N, l, d)`."""
        if self._state is None or self.done:
            raise RuntimeError('Update trajectory of empty state or for step > trajectory steps')

        pred = pred[:, self.current_index_step : self.current_index_step + self.L_min].to(self.dtype).detach()
        time_diff = self.time_diff[self.current_index_step : self.current_index_step + self.L_min]

        self._state = self._state + einsum(time_diff, pred, "t, b t l d -> b l d")
        self._state_count += self.L_min

    @property
    def done(self):
        return self.current_index_step >= self.trajectory_steps

    @property
    def state(self):
        if self._state is None: 
            raise RuntimeError('Attempt to empty trajectory')
        return self._state

    @property
    def current_index_step(self):
        if self._state_count is None: 
            raise RuntimeError('Attempt to empty trajectory')
        return self._state_count


class PDDModule(BaseModule):
    def __init__(
        self,
        trajectory_steps: int,
        L_min: int,
        L_max: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trajectory_steps = trajectory_steps
        self.L_min = L_min
        self.L_max = L_max
        
        if trajectory_steps % L_min:
            raise ValueError(f"trajectory_steps={trajectory_steps} must be a multiple of L_min={L_min}")
        if not L_min <= L_max <= trajectory_steps:
            raise ValueError(f"Need L_min <= L_max <= trajectory_steps, got L_max={L_max}")

    
    def sample_target(self, step_idx: int, generator: torch.Generator):
        r"""Sample a target `k` in `[n, min(n + L_max, N))` for time index `n = step_idx`."""
        hi = min(step_idx + self.L_max, self.trajectory_steps)
        return torch.randint(step_idx, hi, (1,), device=generator.device, generator=generator).item()

    
    @torch.no_grad()
    def pdd_mid_step(
        self,
        x_n,
        pred,
        time_diff,
        step_idx: int,
        generator: torch.Generator | None = None,
        k = None,
    ):

        if k is None:
            if generator is None:
                raise ValueError("pdd_mid_step needs a generator to sample k, or an explicit k")
            k = self.sample_target(step_idx, generator)
        elif not step_idx <= k < min(step_idx + self.L_max, self.trajectory_steps):
            raise ValueError(f"Target k={k} outside [{step_idx}, min({step_idx + self.L_max}, {self.trajectory_steps}))")

        x_k = x_n + einsum(time_diff[step_idx:k], pred[:, step_idx:k].to(x_n.dtype), "t, b t l d -> b l d")
        return x_k, k
