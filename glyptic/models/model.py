from data.data_loader import FrameKeyDataLoader
from glyptic.models.train import load_data
from jax import grad, random, vmap, jit
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import time

input_images = load_data()

input_images.shape


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def linear_diffusion_schedule(num_steps):
    min_rate = 0.0001
    max_rate = 0.02
    betas = min_rate + jnp.linspace(0, 1, num_steps) * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = jnp.cumprod(alphas)
    signal_rates = alpha_bars
    noise_rates = 1 - alpha_bars
    return noise_rates, signal_rates


def cosine_diffusion_schedule(num_steps):
    times = jnp.linspace(0, 1, num_steps)
    signal_rates = jnp.cos(times * (jnp.pi / 2))
    noise_rates = jnp.sin(times * (jnp.pi / 2))
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(num_steps):
    times = jnp.linspace(0, 1, num_steps)
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = jnp.acos(max_signal_rate)
    end_angle = jnp.acos(min_signal_rate)

    diffusion_angles = start_angle + times * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates


def relu(z):
    return jnp.maximum(0, z)


# TODO: Revise predict func, esp. output values (currently probs)
def predict(params, x):
    a = x
    for w, b in params[:-1]:
        z = jnp.dot(w, a) + b
        a = relu(z)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, a) + final_b
    # TODO: Figure out activation func.
    return logits


batched_predict = vmap(predict, in_axes=(None, 0))


# TODO: Batch
def denoise(params, noisy_image, noise_rate, signal_rate):
    # TODO:
    # - find out why square the noise rates
    # - figure out the more optimal method (perhaps array concat more appropriate)
    pred_noise = predict(params, [noisy_image, noise_rate**2])
    # Remove noise scaled by number of noise steps and scale back up by signal rates to overall same
    # intensity
    pred_image = (noisy_image - noise_rate * pred_noise) / signal_rate
    return pred_noise, pred_image


# MAE
def loss(y, preds):
    return jnp.mean(jnp.abs(y - preds))


@jit
# Figure out batching
def update(params, x, y, noise_rate, signal_rate):
    # Step_size = learning_rate??
    step_size = 0.01
    preds = denoise(params, x, noise_rate, signal_rate)
    grads = grad(loss)(y, preds)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]


# Input layer size flattened - determine if suitable
layer_sizes = [1350, 512, 512, 10]
batch_size = 128
params = init_network_params(layer_sizes, random.key(0))

data_loader = FrameKeyDataLoader(
    "data/combined/collection_20240813-195533/", batch_size=128, shuffle=False
)
num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()
    # TODO: Massively fucked up dimensions, fix
    noise_rate, signal_rate = offset_cosine_diffusion_schedule(1)
    for x, y in data_loader:
        params = update(params, x, y, noise_rate, signal_rate)
    epoch_time = time.time() - start_time

    # train_loss = loss(params, train_images, train_labels)
    # test_loss = loss(params, test_images, test_labels)
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    # print(f"Training set accuracy {train_loss}")
    # print(f"Test set accuracy {test_loss}")

