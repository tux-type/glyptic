from typing import Mapping

from flax.core import FrozenDict, freeze
import jax
import jax.numpy as jnp


def create_noise_schedule(method: str, num_steps: int) -> FrozenDict:
    if method == "cosine":
        return create_cosine_noise_schedule(num_steps=num_steps)
    elif method == "linear":
        return create_linear_noise_schedule(num_steps=num_steps)
    else:
        raise RuntimeError(f"{method} is not a valid option for a noise schedule")


# TODO: Review if correct
def create_cosine_noise_schedule(num_steps: int, s=0.008) -> FrozenDict:
    steps = num_steps + 1
    t = jnp.linspace(0, num_steps, steps)
    T = num_steps
    alpha_bar = jnp.cos(((t / T) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    beta = jnp.clip(beta, 0.0001, 0.999)
    # beta = jnp.clip(beta, 0.0001, 0.9999)
    # sigma2 = ((1 - alpha_bar[:-1]) / (1 - alpha_bar[1:])) * beta
    sigma2 = beta
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha, axis=0)
    return freeze(
        {"alpha": alpha, "alpha_bar": alpha_bar, "sigma2": sigma2, "num_steps": num_steps}
    )


def create_linear_noise_schedule(num_steps: int) -> FrozenDict:
    beta = jnp.linspace(0.0001, 0.02, num_steps)
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha, axis=0)
    sigma2 = beta
    return freeze(
        {"alpha": alpha, "alpha_bar": alpha_bar, "sigma2": sigma2, "num_steps": num_steps}
    )


@jax.jit
def q_xt_x0(alpha_bar: jax.Array, x0: jax.Array, times: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Signal rates at different time steps
    alpha_bar = alpha_bar[times].reshape(-1, 1, 1, 1)
    mean = alpha_bar**0.5 * x0
    variance = 1 - alpha_bar
    return mean, variance


@jax.jit
def q_sample(alpha_bar, x0: jax.Array, times: jax.Array, epsilon: jax.Array):
    mean, variance = q_xt_x0(alpha_bar=alpha_bar, x0=x0, times=times)
    return mean + (variance**0.5) * epsilon


def p_sample(
    rng: jax.Array,
    noise_schedule: Mapping[str, jax.Array],
    epsilon_theta,
    xt: jax.Array,
    times: jax.Array,
) -> jax.Array:
    alpha_bar = noise_schedule["alpha_bar"][times].reshape(-1, 1, 1, 1)
    alpha = noise_schedule["alpha"][times].reshape(-1, 1, 1, 1)
    epsilon_coef = (1 - alpha) / ((1 - alpha_bar) ** 0.5)
    # TODO: MISTAKE: used to be (xt - epsilon_coef - epsilon_theta)
    mean = (1 / (alpha**0.5)) * (xt - epsilon_coef * epsilon_theta)
    variance = noise_schedule["sigma2"][times].reshape(-1, 1, 1, 1) ** 0.5
    epsilon = jax.random.normal(rng, shape=xt.shape)
    return mean + variance * epsilon
