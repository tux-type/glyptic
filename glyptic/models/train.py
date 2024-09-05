from jax import value_and_grad, vmap
from jax import random
import jax.numpy as jnp

from .sample import create_noise_schedule, q_sample


def update_step(apply_fn, x0_batch, y_batch, opt_state, params, num_steps):
    # TODO: Decide how to handle keys
    key = random.key(0)
    noise_schedule = create_noise_schedule(num_steps=num_steps)
    batch_size = x0_batch.shape[0]
    times_batch = random.randint(key, shape=(batch_size,), minval=0, maxval=num_steps)
    noise_batch = random.normal(key, shape=x0_batch.shape)
    xt_batch = q_sample(
        key, noise_schedule["alpha_bar"], x0=x0_batch, times=times_batch, epsilon=noise_batch
    )

    def batch_loss(params):
        def loss_fn(x, times, noise):
            epsilon_theta = apply_fn(params, x, times)
            return (noise - epsilon_theta) ** 2  # TODO: Replace with real loss function

        loss = vmap(
            loss_fn,
            # axis_name="batch",  # Name batch dim
        )(xt_batch, times_batch, noise_batch)
        return jnp.mean(loss)

    (loss, updated_state), grads = value_and_grad(batch_loss, has_aux=True)(params)

    # updates, opt_state = tx.update(grads, opt_state)  # Defined below.
    # params = optax.apply_updates(params, updates)
    return opt_state, params, updated_state, loss
