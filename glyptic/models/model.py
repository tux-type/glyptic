from data.data_loader import FrameKeyDataLoader
from glyptic.models.train import load_data
from jax import grad, random, vmap, jit
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import time


# input_images = load_data()
#
# input_images.shape
#
# # Do not reshape for real network - to use conv
# new_input_images = input_images.reshape(input_images.shape[0], -1)


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


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


def relu(z):
    return jnp.maximum(0, z)


# TODO: Revise predict func, esp. output values (currently probs)
@jit
def predict(params, x):
    a = x
    for w, b in params[:-1]:
        z = jnp.dot(w, a) + b
        a = relu(z)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, a) + final_b
    # TODO: Figure out activation func.
    return logits


@jit
def batch_predict(params, batched_x):
    # return vmap(predict, in_axes=(None, 0))(params, batched_x)
    return vmap(predict)(params, batched_x)


@jit
def denoise(params, noisy_image, noise_rate, signal_rate):
    # TODO:
    # - find out why square the noise rates - to have smaller steps for denoising than noising?
    # - figure out the more optimal method (perhaps array concat more appropriate)
    pred_noise = predict(params, noisy_image)
    # REAL IMPLEMENTATION:
    # pred_noise = predict(params, [noisy_image, noise_rate**2])
    # Remove noise scaled by number of noise steps and scale back up by signal rates to overall same
    # intensity
    pred_image = (noisy_image - noise_rate * pred_noise) / signal_rate
    return pred_noise, pred_image


@jit
def batch_denoise(params, batch_noisy_image, batch_noise_rate, batch_signal_rate):
    return vmap(denoise)(params, batch_noisy_image, batch_noise_rate, batch_signal_rate)


@jit
def mean_absolute_error(y, preds):
    return jnp.mean(jnp.abs(y - preds))


@jit
def batch_mean_absolute_error(batch_y, batch_preds):
    return vmap(mean_absolute_error)(batch_y, batch_preds)


# MAE
@jit
def batch_loss(batch_noise_preds, batch_noise):
    return batch_mean_absolute_error(batch_noise_preds, batch_noise)


# TODO: @jit
# Figure out batching
def update(params, batch_x, batch_y):
    # TODO: Figure out a good way to track keys - perhaps split
    noises = random.normal(random.key(1), shape=x.shape)
    learning_rate = 0.01
    batch_size = batch_x.shape[0]
    time_steps = random.uniform(random.key(2), shape=batch_size, minval=0.0, maxval=1.0)
    batch_noise_rate, batch_signal_rate = cosine_diffusion_schedule(time_steps)
    batch_pred_noises, batch_pred_images = batch_denoise(
        params, batch_x, batch_noise_rate, batch_signal_rate
    )
    loss = batch_loss(batch_pred_noises, noises)
    print(f"loss: {loss}")
    grads = grad(batch_loss)(batch_pred_noises, noises)
    return [
        (w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)
    ]


# Input layer size flattened - determine if suitable
layer_sizes = [4050, 512, 512, 4050]
batch_size = 128
params = init_network_params(layer_sizes, random.key(0))

data_loader = FrameKeyDataLoader(
    "data/combined/collection_20240813-195533/", batch_size=128, shuffle=False
)

# random.uniform

num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in data_loader:
        # TODO: Remove reshape - Not needed when using conv later
        params = update(params, x.reshape(x.shape[0], -1), y)
    epoch_time = time.time() - start_time

    # train_loss = loss(params, train_images, train_labels)
    # test_loss = loss(params, test_images, test_labels)
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    # print(f"Training set accuracy {train_loss}")
    # print(f"Test set accuracy {test_loss}")
