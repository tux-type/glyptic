from typing import TypedDict
from flax.training import train_state
from jax import value_and_grad, vmap, random, Array
import jax.numpy as jnp
import optax

from .layers import UNet
from .sample import create_noise_schedule, q_sample
from .config import MyConfig


def create_train_state(rng: Array, config: MyConfig):
    unet = UNet()
    # TODO: Dynamically adjust input size
    params = unet.init(rng, jnp.ones((128, 32, 48, 3)))["params"]
    # TODO: Determine if worth using momentum here
    tx = optax.adam(learning_rate=config.learning_rate)
    return train_state.TrainState.create(apply_fn=unet.apply, params=params, tx=tx)


def apply_model(state: train_state.TrainState, xt_batch, times_batch, noise_batch):
    def batch_loss(params):
        def squared_error(x, times, epsilon):
            epsilon_theta = state.apply_fn(params, x, times)
            return jnp.inner(epsilon - epsilon_theta, epsilon - epsilon_theta) / 2.0

        loss = vmap(
            squared_error,
        )(xt_batch, times_batch, noise_batch)
        # TODO: Check if need to add axis, e.g. axis = 0
        return jnp.mean(loss)

    loss, grads = value_and_grad(batch_loss)(state.params)
    return loss, grads


def train_epoch():
    pass


def update_step(apply_function, x0_batch, opt_state, params, num_steps):
    key = random.key(0)
    noise_schedule = create_noise_schedule(num_steps=num_steps)
    batch_size = x0_batch.shape[0]
    times_batch = random.randint(key, shape=(batch_size,), minval=0, maxval=num_steps)
    noise_batch = random.normal(key, shape=x0_batch.shape)
    xt_batch = q_sample(
        key, noise_schedule["alpha_bar"], x0=x0_batch, times=times_batch, epsilon=noise_batch
    )

    def batch_loss(params):
        def squared_error(x, times, epsilon):
            epsilon_theta = apply_function(params, x, times)
            return jnp.inner(epsilon - epsilon_theta, epsilon - epsilon_theta) / 2.0

        loss = vmap(
            squared_error,
        )(xt_batch, times_batch, noise_batch)
        # TODO: Check if need to add axis, e.g. axis = 0
        return jnp.mean(loss)

    loss, grads = value_and_grad(batch_loss)(params)

    # updates, opt_state = tx.update(grads, opt_state)  # Defined below.
    # params = optax.apply_updates(params, updates)
    # return opt_state, params, loss
