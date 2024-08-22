from glyptic.data import FrameKeyDataLoader
import jax.numpy as jnp
from jax import Array, jit, vmap


def preprocess(image: Array):
    image = image.astype("float32") / 255.0
    return image


@jit
def batched_preprocess(batched_images: Array):
    return vmap(preprocess)(batched_images)


def load_data() -> Array:
    data_loader = FrameKeyDataLoader(
        "data/combined/collection_20240813-195533/", batch_size=1, shuffle=False
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


def train_model():
    pass


if __name__ == "__main__":
    train_model()
