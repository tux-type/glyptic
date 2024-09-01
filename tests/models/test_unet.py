from glyptic.models.layers import (
    AttentionBlock,
    DownBlock,
    DownSample,
    MiddleBlock,
    SinusoidalEmbedding,
    ResidualBlock,
    UpBlock,
    UpSample,
)
import jax.numpy as jnp
from jax import random, Array


def test_sinusoidal_embedding():
    batch_size = 128
    channels = 64
    n_steps = 1000
    key = random.key(0)

    times = random.randint(key, shape=(batch_size,), minval=0, maxval=n_steps)

    se = SinusoidalEmbedding(num_channels=channels * 4)
    params = se.init(key, times)
    outputs = se.apply(params, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 256)


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


def test_attention_block():
    batch_size = 128
    height, width = 30, 45
    channels = 64
    key = random.key(0)

    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)

    attn = AttentionBlock(num_heads=1)
    params = attn.init(key, inputs)
    outputs = attn.apply(params, inputs)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 30, 45, 64)


def test_down_block():
    batch_size = 128
    height, width = 30, 45
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    db = DownBlock(features=128, with_attention=True)
    params = db.init(key, inputs, times)
    outputs = db.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 30, 45, 128)


def test_up_block():
    batch_size = 128
    height, width = 30, 45
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    ub = UpBlock(features=32, with_attention=True)
    params = ub.init(key, inputs, times)
    outputs = ub.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 30, 45, 32)


def test_middle_block():
    batch_size = 128
    height, width = 30, 45
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    mb = MiddleBlock(features=32)
    params = mb.init(key, inputs, times)
    outputs = mb.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 30, 45, 32)


def test_down_sample():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)

    ds = DownSample(channels)
    params = ds.init(key, inputs)
    outputs = ds.apply(params, inputs)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 16, 96, 64)


def test_up_sample():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)

    us = UpSample(channels)
    params = us.init(key, inputs)
    outputs = us.apply(params, inputs)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 64, 96, 64)
