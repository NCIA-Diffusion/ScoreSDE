import abc
import numpy as np
import torch
import torch.nn as nn


class AbstractSDE(abc.ABC):
    def __init__(self):
        super().__init__()
        self.N = 1000

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        raise NotImplementedError

    @abc.abstractmethod
    def sde(self, x_t, t):
        """Compute the drift/diffusion of the forward SDE
        dx = b(x_t, t)dt + s(x_t, t)dW
        """
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_prob(self, x_0, t):
        """Compute the mean/std of the transitional kernel
        p(x_t | x_0).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def scale_start_to_noise(self, t):
        """Compute the scale of conversion
        from the original image estimation loss, i.e, || x_0 - x_0_pred ||
        to the noise prediction loss, i.e, || e - e_pred ||.
        Denoting the output of this function by C, 
        C * || x_0 - x_0_pred || = || e - e_pred || holds.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def proposal_distribution(self):
    #     raise NotImplementedError

    def reverse(self, model, model_pred_type='noise'):
        """The reverse-time SDE."""
        sde_fn = self.sde
        marginal_fn = self.marginal_prob

        class RSDE(self.__class__):
            def __init__(self):
                pass
                
            def sde(self, x_t, t):
                # Get score function values
                if model_pred_type == 'noise':
                    x_noise_pred = model(x_t, t)
                    _, x_std = marginal_fn(
                        torch.zeros_like(x_t),
                        t,
                    ) 
                    score = -x_noise_pred / x_std

                elif model_pred_type == 'original':
                    x_0_pred = model(x_t, t)
                    x_mean, x_std = marginal_fn(
                        x_0_pred,
                        t
                    )
                    score = (x_mean - x_t) / x_std

                # Forward SDE's drift & diffusion
                drift, diffusion = sde_fn(x_t, t)

                # Reverse SDE's drift & diffusion (Anderson, 1982)
                drift = drift - diffusion ** 2 * score
                return drift, diffusion

        return RSDE()


class VPSDE(AbstractSDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        # self.IS_dist, self.norm_const = self.proposal_distribution()

    @property
    def T(self):
        return 1

    def sde(self, x_t, t):
        beta_t = (self.beta_0 + t * (self.beta_1 - self.beta_0))[:, None, None, None]
        drift = -0.5 * beta_t * x_t
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )[:, None, None, None]
        marginal_mean = torch.exp(log_mean_coeff) * x_0
        marginal_std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return marginal_mean, marginal_std

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = - N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def scale_start_to_noise(self, t):
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )[:, None, None, None]
        marginal_coeff = torch.exp(log_mean_coeff)
        marginal_std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        scale = marginal_coeff / (marginal_std + 1e-12)
        return scale

    # def proposal_distribution(self):
    #     def g2(t):
    #         return self.beta_0 + t * (self.beta_1 - self.beta_0)
    #     def a2(t):
    #         log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) \
    #             - 0.5 * t * self.beta_0
    #         return 1. - torch.exp(2. * log_mean_coeff)
    #     t = torch.arange(1, 1001) / 1000
    #     p = g2(t) / a2(t)
    #     normalizing_const = p.sum()
    #     return p, normalizing_const



