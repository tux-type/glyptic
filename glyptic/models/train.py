from typing import Any

from flax.training.train_state import TrainState
from jax import Array, random, value_and_grad, vmap
from jax import jit
import jax.numpy as jnp
import optax

from glyptic.data import FrameKeyDataLoader

from .layers import UNet
from .sample import create_noise_schedule, q_sample


def create_train_state(rng: Array, config: dict[str, Any]):
    # TODO: Replace with external params
    unet = UNet(
        num_channels=config["initial_channels"],
        channel_multipliers=config["channel_multipliers"],
        # channel_multipliers=(1, 2, 2, 4),
        use_attention=config["blocks_with_attention"],
        num_blocks=config["num_blocks"],
    )
    mock_images = jnp.ones(shape=(1, config["image_height"], config["image_width"], 3))
    mock_times = jnp.ones(shape=(1,))
    params = unet.init(rng, mock_images, mock_times)["params"]
    # TODO: Determine if worth passing momentum here
    tx = optax.adam(learning_rate=config["learning_rate"])
    return TrainState.create(apply_fn=unet.apply, params=params, tx=tx)


@jit
def preprocess(image: Array):
    image = image.astype(jnp.float32) / 255.0
    return image


@jit
def apply_model(state: TrainState, xt_batch: Array, times_batch: Array, epsilon_batch: Array):
    grad_function = value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_function(state.params, state, xt_batch, times_batch, epsilon_batch)
    return loss, grads


@jit
# TODO: Determine if need to pass params separately from state for any reason
def loss_fn(
    params: Array, state: TrainState, xt_batch: Array, times_batch: Array, epsilon_batch: Array
):
    epsilon_theta = state.apply_fn({"params": params}, xt_batch, times_batch)
    loss = optax.l2_loss(epsilon_theta, epsilon_batch)
    return jnp.mean(loss)


def train_step(state: TrainState, x0_batch: Array, config: dict[str, Any], rng: Array):
    times_rng, noise_rng = random.split(rng)
    noise_schedule = create_noise_schedule(num_steps=config["num_steps"])
    batch_size = x0_batch.shape[0]
    times_batch = random.randint(
        times_rng, shape=(batch_size,), minval=0, maxval=config["num_steps"]
    )
    noise_batch = random.normal(noise_rng, shape=x0_batch.shape)
    xt_batch = q_sample(
        alpha_bar=noise_schedule["alpha_bar"], x0=x0_batch, times=times_batch, epsilon=noise_batch
    )
    loss, grads = apply_model(
        state=state, xt_batch=xt_batch, times_batch=times_batch, epsilon_batch=noise_batch
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


# TODO: Add loss calculation
def eval_step(state: TrainState, x0_batch: Array, config: dict[str, Any], rng: Array):
    times_rng, noise_rng = random.split(rng)
    noise_schedule = create_noise_schedule(num_steps=config["num_steps"])
    batch_size = x0_batch.shape[0]
    times_batch = random.randint(
        times_rng, shape=(batch_size,), minval=0, maxval=config["num_steps"]
    )
    noise_batch = random.normal(noise_rng, shape=x0_batch.shape)
    xt_batch = q_sample(
        alpha_bar=noise_schedule["alpha_bar"], x0=x0_batch, times=times_batch, epsilon=noise_batch
    )
    loss = loss_fn(
        params=state.params,
        state=state,
        xt_batch=xt_batch,
        times_batch=times_batch,
        epsilon_batch=noise_batch,
    )
    return loss


def train_and_evaluate(config: dict[str, Any]):
    rng = random.key(config["rng_seed"])

    rng, init_rng, eval_rng = random.split(rng, num=3)
    state = create_train_state(rng=init_rng, config=config)

    # Decide how to change data as Array and move it to GPU efficiently
    data_loader = FrameKeyDataLoader(
        config["train_data_dir"], batch_size=config["batch_size"], shuffle=False
    )
    validation_data_loader = FrameKeyDataLoader(
        config["val_data_dir"], batch_size=config["batch_size"], shuffle=False
    )

    for epoch in range(1, config["num_epochs"] + 1):
        training_loss = []
        for x, y in data_loader:
            rng, batch_rng = random.split(rng)
            x = vmap(preprocess)(jnp.array(x))
            state, loss = train_step(state, x0_batch=x, config=config, rng=batch_rng)
            training_loss.append(loss)
        # TODO: Determine if better to reuse RNG for more consistent evals across epochs
        # TODO: Check if better to accumulate and average rather than calculate across whole data set
        # train_set_loss = eval_step(
        #     state, x0_batch=jnp.array(data_loader.all_images[0]), config=config, rng=eval_rng
        # )
        val_set_loss = eval_step(
            state,
            x0_batch=jnp.array(validation_data_loader.all_images[0]),
            config=config,
            rng=eval_rng,
        )
        print(jnp.array(training_loss).mean(), val_set_loss)
