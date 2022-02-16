import os
from datetime import datetime
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, logsnr):
        super().__init__()

        self.model = model
        self.T = T
        self.logsnr = logsnr

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        self.register_buffer(
            'difflogsnr', self.betas/(1-alphas_bar))

    def forward(self, x_0, y_0, input_dropout_opt):
        """
        Algorithm 1.
        """
        # plt.plot(self.difflogsnr.detach().cpu().numpy())
        # plt.savefig('hi2')
        # plt.close()
        # assert 0==1
        if self.logsnr:
            t = self.difflogsnr.multinomial(num_samples=x_0.shape[0], replacement=True).to(x_0.device)
        else:
            t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        x_noise = torch.randn_like(x_0)
        y_noise = torch.randn_like(y_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * x_noise)
        y_t = (
            extract(self.sqrt_alphas_bar, t, y_0.shape) * y_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape) * y_noise)
        
        # x_noise_pred, y_noise_pred = self.model(x_t, y_t, t, input_dropout_opt)
        # x_loss = F.mse_loss(x_noise_pred, x_noise, reduction='mean')
        # y_loss = F.mse_loss(y_noise_pred, y_noise, reduction='mean')
        x_0_pred, y_0_pred = self.model(x_t, y_t, t, input_dropout_opt)
        x_scale = extract(self.sqrt_alphas_bar, t, x_0.shape) / extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
        y_scale = extract(self.sqrt_alphas_bar, t, y_0.shape) / extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape)
        x_loss = F.mse_loss(x_0_pred * x_scale, x_0 * x_scale, reduction='mean')
        y_loss = F.mse_loss(y_0_pred * y_scale, y_0 * y_scale, reduction='mean')
        return x_loss, y_loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, n_corrector, snr,
        img_size=32, mean_type='epsilon', var_type='fixedlarge'):
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.n_corrector = n_corrector
        self.snr = snr
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = self.alphas = 1. - self.betas
        alphas_bar = self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
        )
        self.register_buffer(
            'sqrt_recip_alphas_bar_m1', 1. / torch.sqrt(1. - alphas_bar)
        )
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, y_t, t, x_0=None, y_0=None):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var_x = extract(model_log_var, t, x_t.shape)
        model_log_var_y = extract(model_log_var, t, y_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev

        elif self.mean_type == 'xstart':    # the model predicts x_0
            _x_0, _y_0 = self.model(x_t, y_t, t)
            if x_0 is None:
                x_0 = _x_0
                model_mean_x, _ = self.q_mean_variance(x_0, x_t, t)
            else:
                model_mean_x = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
                model_log_var_x = 2 * torch.log(extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape))
            if y_0 is None:
                y_0 = _y_0
                model_mean_y, _ = self.q_mean_variance(y_0, y_t, t)
            else:
                model_mean_y = extract(self.sqrt_alphas_bar, t, y_0.shape) * y_0
                model_log_var_y = 2 * torch.log(extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape))

        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps_x, eps_y = self.model(x_t, y_t, t)
            if x_0 is None:
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps_x)
                model_mean_x, _ = self.q_mean_variance(x_0, x_t, t)
            else:
                model_mean_x = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0
                model_log_var_x = 2 * torch.log(extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape))
            if y_0 is None:
                y_0 = self.predict_xstart_from_eps(y_t, t, eps=eps_y)
                model_mean_y, _ = self.q_mean_variance(y_0, y_t, t)
            else:
                model_mean_y = extract(self.sqrt_alphas_bar, t, y_0.shape) * y_0
                model_log_var_y = 2 * torch.log(extract(self.sqrt_one_minus_alphas_bar, t, y_0.shape))

        else:
            raise NotImplementedError(self.mean_type)

        return model_mean_x, model_mean_y, model_log_var_x, model_log_var_y

    def predictor(self, time_step, x_t, y_t, x_0, y_0):
        t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
        x_mean, y_mean, x_logvar, y_logvar = self.p_mean_variance(x_t=x_t, y_t=y_t, t=t, x_0=x_0, y_0=y_0)
        
        if time_step > 0:
            x_noise = torch.randn_like(x_t)
            y_noise = torch.randn_like(y_t)
        else:
            x_noise = 0
            y_noise = 0

        x_t = x_mean + torch.exp(0.5 * x_logvar) * x_noise
        y_t = y_mean + torch.exp(0.5 * y_logvar) * y_noise

        return x_t, y_t

    def corrector(self, time_step, x_t, y_t, option='joint_gen'):
        if time_step == 0 or self.n_corrector == 0:
            return x_t, y_t
        t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
        recip_std_x = extract(self.sqrt_recip_alphas_bar_m1, t, x_t.shape)
        recip_std_y = extract(self.sqrt_recip_alphas_bar_m1, t, y_t.shape)
        for _ in range(self.n_corrector):
            g_x, g_y = self.model(x_t, y_t, t)
            g_x = -g_x * recip_std_x
            g_y = -g_y * recip_std_y

            ch_x = g_x.shape[1] # channel of x
            g = torch.cat([g_x, g_y], dim=1) # channel-wise concatenate
            g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=-1).mean()
            z = torch.randn_like(g)
            z_norm = torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean()
            temp = (self.snr * z_norm / g_norm) ** 2

            if option != 'cls':
                a_x = extract(self.alphas.to(x_t.device), t, x_t.shape)
                e_x = 2 * a_x * temp
                x_t = x_t + e_x * g_x + (2 * e_x).sqrt() * z[:, :ch_x]

            if option != 'cond_gen':
                a_y = extract(self.alphas.to(y_t.device), t, y_t.shape)
                e_y = 2 * a_y * temp
                y_t = y_t + e_y * g_y + (2 * e_y).sqrt() * z[:, ch_x:]

        return x_t, y_t

    def forward(self, x_T, y_T, x_0=None, y_0=None, option='joint_gen', logdir=None):
        """
        Algorithm 2.
        """
        # Assertion
        if x_0 is None and y_0 is None:
            assert option == 'joint_gen'
        elif x_0 is not None:
            assert option == 'cls'
        elif y_0 is not None:
            assert option == 'cond_gen'
        else:
            raise ValueError

        if option == 'joint_gen':
            # Backward (ONLY)
            x_t = x_T
            y_t = y_T
            T = self.T
            for time_step in reversed(range(T)):

                # Predictor
                x_t, y_t = self.predictor(time_step, x_t, y_t, x_0, y_0)

                # Corrector (used only if t > 0)
                x_t, y_t = self.corrector(time_step, x_t, y_t, option=option)

            x_0 = x_t
            y_0 = y_t
            return torch.clip(x_0, -1., 1.), torch.clip(y_0, -1., 1.)

        elif option == 'cond_gen' or option == 'cls':

            for k in range(1):
                # Forward
                if x_0 is not None:
                    x_t = x_0
                    y_t = torch.zeros_like(y_T, device=y_T.device)
                elif y_0 is not None:
                    x_t = torch.zeros_like(x_T, device=x_T.device)
                    y_t = y_0
                else:
                    raise ValueError

                T = int(self.T - 1) # 999
                tensor_T = x_t.new_ones([x_T.shape[0],], dtype=torch.long) * T
                x_t = extract(self.sqrt_alphas_bar, tensor_T, x_t.shape) * x_t
                y_t = extract(self.sqrt_alphas_bar, tensor_T, y_t.shape) * y_t
                x_noise = torch.randn_like(x_t)
                y_noise = torch.randn_like(y_t)
                x_t = x_t + extract(self.sqrt_one_minus_alphas_bar, tensor_T, x_t.shape) * x_noise
                y_t = y_t + extract(self.sqrt_one_minus_alphas_bar, tensor_T, y_t.shape) * y_noise

                # for name, img in {'image': x_t, 'label': y_t}.items():
                #     img = (torch.clip(img, -1, 1) / 2 + 1 / 2).detach().cpu()
                #     now = datetime.now()
                #     save_image(
                #         img[:64],
                #         os.path.join(logdir, f'START_REVERSE_{name}_{k}_{now}.png'),
                #         nrow=8,
                #     )

                # Backward
                for time_step in reversed(range(T)):

                    # Predictor
                    x_t, y_t = self.predictor(time_step, x_t, y_t, x_0, y_0)

                    # Corrector (used only if t > 0)
                    x_t, y_t = self.corrector(time_step, x_t, y_t, option=option)

                # for name, img in {'image': x_t, 'label': y_t}.items():
                #     img = (torch.clip(img, -1, 1) / 2 + 1 / 2).detach().cpu()
                #     now = datetime.now()
                #     save_image(
                #         img[:64],
                #         os.path.join(logdir, f'END_REVERSE_{name}_{k}_{now}.png'),
                #         nrow=8,
                #     )

            x_0 = x_t
            y_0 = y_t
            return torch.clip(x_0, -1, 1), torch.clip(y_0, -1, 1)

