from glyptic.models.layers import ResidualBlock
import jax.numpy as jnp
from jax import random, Array


def test_residual_block():
    batch_size = 128
    height, width = 30, 45
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    rb = ResidualBlock(filters=32, num_groups=32)
    params = rb.init(key, inputs, times)
    outputs = rb.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 30, 45, 32)
