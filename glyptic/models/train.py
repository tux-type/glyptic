from glyptic.data import FrameKeyDataLoader
import jax.numpy as jnp
from jax import Array, jit, vmap, random, grad
from .model import init_network_params
from .sampling import batch_denoise, cosine_diffusion_schedule

import time


@jit
def preprocess(image: Array):
    image = image.astype(jnp.float32) / 255.0
    return image


@jit
def batched_preprocess(batched_images: Array):
    return vmap(preprocess)(batched_images)


def load_data() -> Array:
    data_loader = FrameKeyDataLoader(
        "data/combined/collection_20240813-195533/", batch_size=128, shuffle=False
    )

    input_images = []
    label_images = []

    for input_image, label_image in data_loader:
        input_images.append(input_image)
        label_images.append(label_image)
        break

    input_images = jnp.array(input_images[0])
    input_images = batched_preprocess(input_images)
    return input_images


@jit
def mean_absolute_error(y, preds):
    return jnp.mean(jnp.abs(y - preds))


# MAE
@jit
def loss(params, batch_x, batch_noise_rate, batch_signal_rate, batch_noise):
    batch_noise_preds, batch_image_preds = batch_denoise(
        params, batch_x, batch_noise_rate, batch_signal_rate
    )
    return mean_absolute_error(batch_noise_preds, batch_noise)


# TODO: @jit
def update(params, batch_x, batch_y, key):
    key_noise, key_time_step = random.split(key)
    noises = random.normal(key_noise, shape=batch_x.shape)
    learning_rate = 0.01
    batch_size = batch_x.shape[0]
    time_steps = random.uniform(key_time_step, shape=batch_size, minval=0.0, maxval=1.0)
    batch_noise_rate, batch_signal_rate = cosine_diffusion_schedule(time_steps)
    loss_i = loss(params, batch_x, batch_noise_rate, batch_signal_rate, noises)
    print(f"loss: {loss_i}")
    grads = grad(loss)(params, batch_x, batch_noise_rate, batch_signal_rate, noises)
    return [
        (w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)
    ]


def train_model():
    # Input layer size flattened - determine if suitable
    layer_sizes = [4050, 512, 512, 4050]
    batch_size = 128
    params_key, update_key = random.split(random.key(0))
    params = init_network_params(layer_sizes, params_key)

    data_loader = FrameKeyDataLoader(
        "data/combined/collection_20240813-195533/", batch_size=batch_size, shuffle=False
    )

    num_epochs = 5
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in data_loader:
            x = batched_preprocess(x)
            update_key, batch_update_key = random.split(update_key)
            # TODO: Remove reshape - Not needed when using conv later
            params = update(params, x.reshape(x.shape[0], -1), y, batch_update_key)
        epoch_time = time.time() - start_time

        # train_loss = loss(params, train_images, train_labels)
        # test_loss = loss(params, test_images, test_labels)
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        # print(f"Training set accuracy {train_loss}")
        # print(f"Test set accuracy {test_loss}")


if __name__ == "__main__":
    train_model()
