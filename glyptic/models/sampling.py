from jax import jit, vmap
import jax.numpy as jnp

from .model import forward


@jit
def linear_diffusion_schedule(time_steps):
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + time_steps * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = jnp.cumprod(alphas)
    signal_rates = alpha_bars
    noise_rates = 1 - alpha_bars
    return noise_rates, signal_rates


@jit
def cosine_diffusion_schedule(time_steps):
    signal_rates = jnp.cos(time_steps * (jnp.pi / 2))
    noise_rates = jnp.sin(time_steps * (jnp.pi / 2))
    return noise_rates, signal_rates


@jit
def offset_cosine_diffusion_schedule(time_steps):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = jnp.acos(max_signal_rate)
    end_angle = jnp.acos(min_signal_rate)

    diffusion_angles = start_angle + time_steps * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates


@jit
def denoise(params, noisy_image, noise_rate, signal_rate):
    # TODO:
    # - find out why square the noise rates - to have smaller steps for denoising than noising?
    # - figure out the more optimal method (perhaps array concat more appropriate)
    pred_noise = forward(params, noisy_image)
    # TODO: REAL IMPLEMENTATION:
    # pred_noise = predict(params, [noisy_image, noise_rate**2])
    # Remove noise scaled by number of noise steps and scale back up by signal rates to overall same
    # intensity
    pred_image = (noisy_image - noise_rate * pred_noise) / signal_rate
    return pred_noise, pred_image


@jit
def batch_denoise(params, batch_noisy_image, batch_noise_rate, batch_signal_rate):
    # in_axes is making sure that params are not included in batching??
    return vmap(denoise, in_axes=(None, 0, 0, 0))(
        params, batch_noisy_image, batch_noise_rate, batch_signal_rate
    )
