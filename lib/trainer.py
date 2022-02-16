import torch
import torch.nn as nn

from .sde import AbstractSDE


class DiffusionTrainer(nn.Module):
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
        assert model_pred_type in ['noise', 'original']
        self.model_pred_type = model_pred_type
        self.t_eps = t_eps
            
    def forward(
        self,
        x_0: torch.Tensor,
    ):
        B = x_0.shape[0]
        device = x_0.device

        # Forward process
        t = torch.rand(B, device=device) * (self.sde.T - self.t_eps) + self.t_eps
        x_noise = torch.randn_like(x_0)
        x_mean, x_std = self.sde.marginal_prob(x_0, t)
        x_t = x_mean + x_std * x_noise

        if self.model_pred_type == 'noise':
            x_noise_pred = self.model(x_t, t)
            loss = torch.square(x_noise_pred - x_noise)
            loss = loss.reshape(B, -1)

        elif self.model_pred_type == 'original':
            x_0_pred = self.model(x_t, t)
            scale = self.sde.scale_start_to_noise(t)
            scale = torch.clip(scale, min=None, max=5.0)
            loss = torch.square((x_0_pred - x_0) * scale)
            loss = loss.reshape(B, -1)

        return loss

        
