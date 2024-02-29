import torch 

class Sampler:
    def __init__(self, g: torch.Generator, steps=1000, b_start: float = 0.00085, b_end: float = 0.0120):
        self.g      = g
        self.betas  = torch.linspace(b_start ** 0.5, b_end ** 0.5, steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.one    = torch.tensor(1.0)
        self.timesteps  = torch.arange(steps-1, -1, -1)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_train_timesteps = steps
        
    def set_ineference_steps(self, num_inference_steps: int=50):
        self.num_inference_steps = num_inference_steps
        step_ratio               = self.num_train_timesteps // self.num_inference_steps
        self.timesteps           = (torch.arange(num_inference_steps-1, -1, -1) * step_ratio).round()

    def _get_prev_step(self, step: int) -> int:
        return step - self.num_inference_steps // self.num_inference_steps
    
    def _get_std(self, step: int) -> torch.Tensor:
        prev_t            = self._get_prev_step(step)
        alpha_prod_t      = self.alphas_cumprod[step]
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t    = 1 - alpha_prod_t / alpha_prod_prev_t
        # compute predicted variance (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf
        var = (1-alpha_prod_t) / (1-alpha_prod_prev_t) * current_beta_t
        return var.clamp(min=1e-20).sqrt()
    
    def set_strength(self, strength: float = 1.):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. remove noise,  x_0 (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 6. Add noise
        std = 0 
        if t > 0:
            noise = torch.randn(model_output.shape, generator=self.g, device=model_output.device, dtype=model_output.dtype)
            std = self._get_std(t)*noise
        
        pred_prev_sample = pred_prev_sample + std
        return pred_prev_sample
    
    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    



