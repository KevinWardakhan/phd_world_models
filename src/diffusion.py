"""DDPM (Ho et al., 2020) implementation for 2D data.

Uses denoising score matching loss and a linear noise schedule
"""

import torch


def linear_beta_schedule(num_timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, num_timesteps)


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        betas = linear_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    @property
    def config(self):
        return {
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }

    @staticmethod
    def _gather(arr, t):
        """Index a schedule array by per-sample timestep, shaped for broadcast."""
        return arr[t].unsqueeze(-1)

    def q_sample(self, x0, t, eps):
        """Forward diffusion: noise x0 to level t."""
        return (
            self._gather(self.sqrt_alphas_cumprod, t) * x0
            + self._gather(self.sqrt_one_minus_alphas_cumprod, t) * eps
        )

    def loss(self, model, x0):
        """Simplified DDPM training loss on a batch of clean samples."""
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        eps_pred = model(x_t, t)
        return ((eps - eps_pred) ** 2).mean()

    def predict_x0(self, x_t, t, eps):
        """Estimate x0 from x_t and predicted noise (Eq. 15)."""
        return (
            x_t - self._gather(self.sqrt_one_minus_alphas_cumprod, t) * eps
        ) / self._gather(self.sqrt_alphas_cumprod, t)

    def eps_to_score(self, eps, t):
        """Score of the diffused distribution from an eps prediction.

        For x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps, the score is
        s(x_t, t) = -eps_theta(x_t, t) / sigma_t with sigma_t = sqrt(1-abar_t).
        Used by DMD (M3) for both the real teacher score and the fake score.
        """
        return -eps / self._gather(self.sqrt_one_minus_alphas_cumprod, t)

    @torch.no_grad()
    def onestep_x0(self, model, x_T):
        """M2 naive one-step baseline: a single x0 estimate from noise (Eq. 15).
        This sampling is deterministic because no noise is sampled.

        One teacher eps prediction at the largest timestep, turned directly into
        x0_hat. This is the diagnostic baseline, NOT a DMD/DMD2 student.
        """
        n = x_T.shape[0]
        t = torch.full((n,), self.num_timesteps - 1, device=self.device, dtype=torch.long)
        eps = model(x_T, t)
        return self.predict_x0(x_T, t, eps)

    @torch.no_grad()
    def p_sample_step(self, model, x_t, t):
        """One reverse step x_t -> x_{t-1} (Alg. 2)."""
        betas_t = self._gather(self.betas, t)
        alphas_t = self._gather(self.alphas, t)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_cumprod, t)

        eps_pred = model(x_t, t)
        mean = (x_t - betas_t / sqrt_one_minus * eps_pred) / torch.sqrt(alphas_t)

        nonzero = (t > 0).float().unsqueeze(-1)
        noise = torch.randn_like(x_t)
        return mean + nonzero * torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, n=None, dim=2, return_trajectory=False, x_init=None):
        """Full reverse process from N(0, I) noise to samples.
        Mimics algorithm 2 in the DDPM paper.

        Pass `x_init` to start from a fixed x_T (so the multi-step teacher and the
        one-step baseline share the same noise); otherwise draws fresh N(0, I).
        Trajectory (when requested) is a list of CPU tensors:
        index 0 is x_T (noise), index j is x_{T-j}, last is x_0.
        """
        x = x_init if x_init is not None else torch.randn(n, dim, device=self.device)
        n = x.shape[0]
        trajectory = [x.detach().cpu().clone()]
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            x = self.p_sample_step(model, x, t)
            if return_trajectory:
                trajectory.append(x.detach().cpu().clone())
        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model, n=None, dim=2, x_init=None):
        """Deterministic DDIM sampler (eta=0) over the full schedule.

        x_{t-1} = sqrt(abar_{t-1}) x0_hat + sqrt(1-abar_{t-1}) eps_pred, ending at
        x0_hat. Deterministic, so it gives reproducible noise->sample pairs for the
        M3 DMD regression dataset.
        """
        x = x_init if x_init is not None else torch.randn(n, dim, device=self.device)
        n = x.shape[0]
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            eps = model(x, t)
            x0 = self.predict_x0(x, t, eps)
            if i > 0:
                abar_prev = self.alphas_cumprod[i - 1]
                x = torch.sqrt(abar_prev) * x0 + torch.sqrt(1.0 - abar_prev) * eps # deterministic sampling
            else:
                x = x0
        return x


def load_teacher(path, device="cpu", which="eval"):
    """Reload a frozen teacher (model + diffusion) from a checkpoint.

    which:
      "eval" -> EMA weights if EMA was used for evaluation, else raw (default,
                this is the frozen teacher M2-M4 should use),
      "ema"  -> EMA weights,
      "raw"  -> raw (non-EMA) weights.
    """
    from .models import MLPDenoiser

    ckpt = torch.load(path, map_location=device)
    use_ema = which == "ema" or (which == "eval" and ckpt.get("ema_used_for_eval"))
    state = ckpt["ema_state_dict"] if use_ema else ckpt["model_state_dict"]

    model = MLPDenoiser.from_config(ckpt["model_config"]).to(device)
    model.load_state_dict(state)
    model.eval()
    diffusion = GaussianDiffusion(device=device, **ckpt["diffusion_config"])
    return model, diffusion
