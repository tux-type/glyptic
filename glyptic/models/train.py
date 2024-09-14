from typing import Any
from datetime import datetime

from clearml import Logger, Task
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from glyptic.data import FrameKeyDataLoader
from glyptic.models.config import get_config
from glyptic.models.layers import UNet
from glyptic.models.sample import create_noise_schedule, q_sample


def create_train_state(rng: jax.Array, config: dict[str, Any]):
    unet = UNet(
        num_channels=config["initial_channels"],
        channel_multipliers=config["channel_multipliers"],
        use_attention=config["blocks_with_attention"],
        num_blocks=config["num_blocks"],
    )
    mock_images = jnp.ones(shape=(1, config["image_height"], config["image_width"], 3))
    mock_times = jnp.ones(shape=(1,))
    params = unet.init(rng, mock_images, mock_times)["params"]
    tx = optax.adam(learning_rate=config["learning_rate"])
    return TrainState.create(apply_fn=unet.apply, params=params, tx=tx)


@jax.jit
def preprocess(image: jax.Array):
    image = image.astype(jnp.float32) / 255.0
    return image


@jax.jit
def apply_model(
    state: TrainState, xt_batch: jax.Array, times_batch: jax.Array, epsilon_batch: jax.Array
):
    grad_function = jax.value_and_grad(loss_fn, argnums=0)
    loss, grads = grad_function(state.params, state, xt_batch, times_batch, epsilon_batch)
    return loss, grads


@jax.jit
def loss_fn(
    params: jax.Array,
    state: TrainState,
    xt_batch: jax.Array,
    times_batch: jax.Array,
    epsilon_batch: jax.Array,
) -> jax.Array:
    epsilon_theta = state.apply_fn({"params": params}, xt_batch, times_batch)
    loss = optax.l2_loss(epsilon_theta, epsilon_batch)
    return jnp.mean(loss)


def train_step(
    state: TrainState, x0_batch: jax.Array, config: dict[str, Any], rng: jax.Array
) -> tuple[TrainState, jax.Array, dict]:
    times_rng, noise_rng = jax.random.split(rng)
    noise_schedule = create_noise_schedule(num_steps=config["num_steps"])
    batch_size = x0_batch.shape[0]
    times_batch = jax.random.randint(
        times_rng, shape=(batch_size,), minval=0, maxval=config["num_steps"]
    )
    noise_batch = jax.random.normal(noise_rng, shape=x0_batch.shape)
    xt_batch = q_sample(
        alpha_bar=noise_schedule["alpha_bar"], x0=x0_batch, times=times_batch, epsilon=noise_batch
    )
    loss, grads = apply_model(
        state=state, xt_batch=xt_batch, times_batch=times_batch, epsilon_batch=noise_batch
    )
    state = state.apply_gradients(grads=grads)
    return state, loss, grads


def eval_step(
    state: TrainState, x0_batch: jax.Array, config: dict[str, Any], rng: jax.Array
) -> jax.Array:
    times_rng, noise_rng = jax.random.split(rng)
    noise_schedule = create_noise_schedule(num_steps=config["num_steps"])
    batch_size = x0_batch.shape[0]
    times_batch = jax.random.randint(
        times_rng, shape=(batch_size,), minval=0, maxval=config["num_steps"]
    )
    noise_batch = jax.random.normal(noise_rng, shape=x0_batch.shape)
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
    batch_rng = jax.random.key(config["rng_seed"])

    train_rng, eval_rng, init_rng = jax.random.split(batch_rng, num=3)
    state = create_train_state(rng=init_rng, config=config)

    # Decide how to change data as jax.Array and move it to GPU efficiently
    data_loader = FrameKeyDataLoader(
        config["train_data_dir"], batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    eval_data_loader = FrameKeyDataLoader(
        config["val_data_dir"], batch_size=config["batch_size"], shuffle=True, drop_last=True
    )

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        for batch_i, (x, y) in enumerate(data_loader):
            train_rng, batch_rng = jax.random.split(train_rng)
            x = jax.vmap(preprocess)(jnp.array(x))
            state, loss, grads = train_step(state, x0_batch=x, config=config, rng=batch_rng)
            iteration = (batch_i + 1) * len(x)
            # TODO: Add logging interval
            Logger.current_logger().report_scalar(
                title="train",
                series="mse_loss",
                value=float(np.mean(loss)),
                iteration=iteration,
            )
            Logger.current_logger().report_scalar(
                title="train",
                series="grads_fro",
                # Frobenius norm
                value=float(jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))),
                iteration=iteration,
            )
            print(f"loss: {loss:>7f}  [{iteration:>5d}/{data_loader.num_samples:>5d}]")

        # Evaluate
        # -------------------------------------------
        validation_loss = []
        for eval_x, eval_y in eval_data_loader:
            eval_rng, batch_rng = jax.random.split(eval_rng)
            val_x = jax.vmap(preprocess)(jnp.array(eval_x))
            eval_loss = eval_step(
                state,
                x0_batch=val_x,
                config=config,
                rng=batch_rng,
            )
            validation_loss.append(eval_loss)

        Logger.current_logger().report_scalar(
            title="validation",
            series="mse_loss",
            value=float(np.mean(validation_loss)),
            iteration=epoch,
        )
    # TODO: Save model


def main():
    jax.config.update("jax_debug_nans", True)
    today = datetime.today().strftime("%Y%m%d")
    task: Task = Task.init(project_name="glyptic", task_name="experiment_lowerish_lr" + today)
    # Logger.set_reporting_nan_value()
    config = get_config()
    task.connect(config)
    train_and_evaluate(config)


if __name__ == "__main__":
    main()
