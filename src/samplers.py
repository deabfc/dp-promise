import torch

from src.schedules import linear_beta_schedule


class DDIMSampler:

    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ) -> None:
        self.img_ch = config.data.img_ch
        self.img_size = config.data.img_size
        self.batch_size = config.sample.batch_size
        self.rho = config.sample.rho
        self.samplesteps = config.sample.samplesteps
        self.num_classes = config.data.num_classes
        self.timesteps = config.diffusion.timesteps
        self.guide_scale = config.sample.guide_scale

        self.betas = linear_beta_schedule(
            timesteps=config.diffusion.timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        )

        c = self.timesteps // self.samplesteps
        self.seq = list(range(0, self.timesteps, c))

        self.betas = torch.cat([torch.zeros(1), self.betas])
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    @torch.no_grad()
    def sample(self, model, shape, y=None):
        device = next(model.parameters()).device
        x = torch.randn(shape, device=device)
        batch_size = shape[0]
        seq_next = [-1] + self.seq[:-1]
        for step, next_step in zip(reversed(self.seq), reversed(seq_next)):
            t = torch.full(
                (batch_size, ),
                step,
                device=device,
                dtype=torch.long,
            )
            x_prev = self.p_sample(model, x, t, step, next_step, y=y)
            x = x_prev
        return x.detach().cpu()

    @torch.no_grad()
    def p_sample(self, model, x, t, step, next_step, y=None):
        alpha_t = self.alphas_cumprod[step + 1]
        alpha_t_next = self.alphas_cumprod[next_step + 1]
        condition_pred_noise = model(x, t, y)
        if self.guide_scale:
            uncondition_pred_noise = model(
                x, t, torch.zeros_like(y).to(y.device))
            pred_noise = (1 + self.guide_scale) * condition_pred_noise - \
                self.guide_scale * uncondition_pred_noise
        else:
            pred_noise = condition_pred_noise
        pred_x0 = (x - pred_noise * (1. - alpha_t).sqrt()) / alpha_t.sqrt()
        sigma = self.rho * ((1. - alpha_t / alpha_t_next) *
                            (1. - alpha_t_next) / (1. - alpha_t)).sqrt()
        noise = torch.randn_like(x)
        dir_xt = (1. - alpha_t_next - sigma**2).sqrt() * pred_noise
        x_prev = alpha_t_next.sqrt() * pred_x0 + dir_xt + sigma * noise
        return x_prev
