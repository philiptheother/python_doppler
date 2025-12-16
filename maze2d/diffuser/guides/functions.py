import pdb
import torch

from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)

def n_step_fps_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True, **kwargs
):
    x_prev = x.detach_().requires_grad_()
    with torch.enable_grad():
        x_0_hat = model.predict_start_from_noise(x_prev, t=t, noise=model.model(x_prev, cond, t))
        if model.clip_denoised:
            x_0_hat.clamp_(-1., 1.)
        else:
            raise RuntimeError()
        
        x_0_hat_cond = apply_conditioning(x_0_hat, cond, model.action_dim)
        y, grad = guide.gradients(x_prev, x_0_hat_cond)

    with torch.no_grad():     
        model_mean, _, model_log_variance = model.q_posterior(x_start=x_0_hat, x_t=x_prev, t=t)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)
        # no noise when t == 0
        noise = torch.randn_like(x_prev)
        noise[t == 0] = 0
        x_t = model_mean + model_std * noise
        x_t = apply_conditioning(x_t, cond, model.action_dim)

        if scale_grad_by_std:
            grad = model_var * grad
        grad[t < t_stopgrad] = 0
        for _ in range(n_guide_steps):
            x_t = x_t + scale * grad
        x_t = apply_conditioning(x_t, cond, model.action_dim)

    return x_t.detach_(), y
