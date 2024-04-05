import torch
import torch.nn.functional as F
import random

from src.schedules import linear_beta_schedule


def extract(a, t, x_shape):
    out = torch.gather(a, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class VanillaDDPMTrainer(torch.nn.Module):

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.timesteps = config.diffusion.timesteps

        self.register_buffer("betas", linear_beta_schedule(
            timesteps=config.diffusion.timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        ).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def forward(self, model, x_start, y=None):
        batch_size = x_start.shape[0]
        device = x_start.device
        t = torch.randint(self.timesteps, size=(batch_size, ), device=device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        pred_noise = model(x_noisy, t, y)

        loss = F.mse_loss(noise, pred_noise)
        return loss


class DPPromiseTrainer(torch.nn.Module):

    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.timesteps = config.diffusion.timesteps
        self.num_noise_sample = config.train.num_noise_sample
        self.S = config.dp.S

        self.register_buffer("betas", linear_beta_schedule(
            timesteps=config.diffusion.timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        ).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def forward(self, model, x_start, y=None, phase="1"):
        batch_size = x_start.shape[0]
        device = x_start.device

        if phase == "1":
            t = torch.randint(
                self.S,
                self.timesteps,
                size=(batch_size, ),
                device=device,
            )
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start, t, noise)
            pred_noise = model(x_noisy, t, y)
            loss = F.mse_loss(noise, pred_noise)
        elif phase == "2":
            x_repeated = x_start.unsqueeze(1).repeat_interleave(
                self.num_noise_sample,
                dim=1,
            )
            x_repeated = x_repeated.reshape(
                batch_size * self.num_noise_sample,
                x_start.shape[1],
                x_start.shape[2],
                x_start.shape[3],
            )

            if y is not None:
                y = y.unsqueeze(1).repeat_interleave(
                    self.num_noise_sample,
                    dim=1,
                )
                y = y.reshape(batch_size * self.num_noise_sample, )

            t = torch.randint(
                self.S,
                size=(batch_size * self.num_noise_sample, ),
                device=device,
            )
            noise = torch.randn_like(x_repeated)
            x_noisy = self.q_sample(x_repeated, t, noise)

            pred_noise = model(x_noisy, t, y)
            loss = F.mse_loss(noise, pred_noise)

        return loss


class DPDiffusionTrainer(torch.nn.Module):

    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.timesteps = config.diffusion.timesteps
        self.num_noise_sample = config.train.num_noise_sample

        self.w1 = config.train.w1
        self.w2 = config.train.w2
        self.w3 = config.train.w3

        self.l1 = config.train.l1
        self.l2 = config.train.l2
        self.l3 = config.train.l3
        self.l4 = config.train.l4

        self.register_buffer("betas", linear_beta_schedule(
            timesteps=config.diffusion.timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        ).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def forward(self, model, x_start, y=None):
        batch_size = x_start.shape[0]
        device = x_start.device

        x_repeated = x_start.unsqueeze(1).repeat_interleave(
            self.num_noise_sample,
            dim=1,
        )
        x_repeated = x_repeated.reshape(
            batch_size * self.num_noise_sample,
            x_start.shape[1],
            x_start.shape[2],
            x_start.shape[3],
        )

        if y is not None:
            y = y.unsqueeze(1).repeat_interleave(
                self.num_noise_sample,
                dim=1,
            )
            y = y.reshape(batch_size * self.num_noise_sample, )

        if random.random() < self.w1:
            t = torch.randint(
                self.l1,
                self.l2,
                size=(batch_size * self.num_noise_sample, ),
                device=device,
            )
        elif random.random() < self.w1 + self.w2:
            t = torch.randint(
                self.l2,
                self.l3,
                size=(batch_size * self.num_noise_sample, ),
                device=device,
            )
        else:
            t = torch.randint(
                self.l3,
                self.l4,
                size=(batch_size * self.num_noise_sample, ),
                device=device,
            )

        noise = torch.randn_like(x_repeated)
        x_noisy = self.q_sample(x_repeated, t, noise)

        pred_noise = model(x_noisy, t, y)
        loss = F.mse_loss(noise, pred_noise)

        return loss
