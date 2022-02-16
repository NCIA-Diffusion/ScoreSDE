from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from .sde import AbstractSDE


class DiffusionSampler(nn.Module):
    def __init__(
        self,
        sde: AbstractSDE,
        model,
        model_pred_type: str = 'noise',
        t_eps: float = 1e-5,
    ):
        """Constructor.
        
        Args:
            sde: Reference SDE for x.
            model: Score function model.
            model_pred_type: Type of the outputs of the model.
                If 'noise', the model predicts noise (eps_x).
                If 'original', the model predicts the original inputs (x_0).
            t_eps: Start-time in SDE.
                Defaults to 1e-5.
        """
        super().__init__()
        self.sde = sde
        self.model = model
        self.rsde = sde.reverse(model, model_pred_type)
        assert model_pred_type in ['noise', 'original']
        self.model_pred_type = model_pred_type
        self.t_eps = t_eps

    @property
    def device(self):
        return next(self.parameters()).device

    def predictor(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        no_noise: bool = False,
    ):
        # Euler-Maruyama
        dt = 1. / self.sde.N
        z = torch.randn_like(x_t)
        drift, diffusion = self.rsde.sde(x_t, t)
        x_prev = x_t - drift * dt
        if not no_noise:
            x_prev += diffusion * np.sqrt(dt) * z
        return x_prev

    def corrector(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        no_noise: bool = False,
    ):
        # Empty yet! 
        return x_t

    def forward(
        self,
        x_T: torch.Tensor,
        pbar: Optional[tqdm] = None,
    ):
        B = x_T.shape[0]
        device = x_T.device
        x_t = x_T
        timesteps = torch.linspace(self.t_eps, self.sde.T, self.sde.N)

        for time_step in reversed(range(self.sde.N)):

            # Set continuous time t (in [0, 1])
            t = timesteps[time_step]
            t = t * torch.ones(B).to(device)

            # Last timestep -> no noise (only drift term is applied)
            if time_step > 0:
                no_noise = False
            else:
                no_noise = True

            # Predictor
            x_t = self.predictor(t, x_t, no_noise)

            # Corrector
            x_t = self.corrector(t, x_t, no_noise)

            # (Optional) show the current time step via tqdm
            if pbar is not None:
                pbar.set_postfix_str(f'Sampling {time_step}')

        x_0 = x_t
        return torch.clip(x_0, -1., 1.)


