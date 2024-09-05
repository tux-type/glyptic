from typing import Mapping
from jax import Array, random
import jax.numpy as jnp
from flax.core import FrozenDict, freeze


def create_noise_schedule(num_steps: int) -> FrozenDict:
    beta = jnp.linspace(0.0001, 0.02, num_steps)
    alpha = 1 - beta
    alpha_bar = jnp.cumprod(alpha, axis=0)
    sigma2 = beta
    return freeze(
        {"alpha": alpha, "alpha_bar": alpha_bar, "sigma2": sigma2, "num_steps": num_steps}
    )


# TODO: Decide whether to jit or not to jit (and/or vmap)
def q_xt_x0(alpha_bar: Array, x0: Array, times: Array) -> tuple[Array, Array]:
    # Signal rates at different times steps
    alpha_bar = alpha_bar[times].reshape(-1, 1, 1, 1)
    mean = alpha_bar**0.5 * x0
    variance = 1 - alpha_bar
    return mean, variance


def q_sample(key: Array, alpha_bar, x0: Array, times: Array, epsilon: Array):
    mean, variance = q_xt_x0(alpha_bar=alpha_bar, x0=x0, times=times)

    return mean + (variance**0.5) * epsilon


def p_sample(
    key,
    noise_schedule: Mapping[str, Array],
    epsilon_theta,
    xt: Array,
    times: Array,
):
    alpha_bar = noise_schedule["alpha_bar"][times].reshape(-1, 1, 1, 1)
    alpha = noise_schedule["alpha"][times].reshape(-1, 1, 1, 1)
    epsilon_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
    mean = 1 / (alpha**0.5) * (xt - epsilon_coef - epsilon_theta)
    variance = noise_schedule["sigma2"][times].reshape(-1, 1, 1, 1)
    epsilon = random.normal(key, shape=xt.shape)
    return mean + (variance**0.5) * epsilon
