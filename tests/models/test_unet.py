from jax import Array, random
import jax.numpy as jnp

from glyptic.models.layers import (
    AttentionBlock,
    DownBlock,
    DownSample,
    MiddleBlock,
    ResidualBlock,
    SinusoidalEmbedding,
    UNet,
    UpBlock,
    UpSample,
)


def test_sinusoidal_embedding():
    batch_size = 128
    channels = 64
    num_steps = 1000
    key = random.key(0)

    times = random.randint(key, shape=(batch_size,), minval=0, maxval=num_steps)

    se = SinusoidalEmbedding(num_channels=channels * 4)
    params = se.init(key, times)
    outputs = se.apply(params, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (128, 256)


def test_residual_block():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    out_channels = 32
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    rb = ResidualBlock(filters=out_channels, num_groups=32)
    params = rb.init(key, inputs, times)
    outputs = rb.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, height, width, out_channels)


def test_attention_block():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    key = random.key(0)

    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)

    attn = AttentionBlock(num_heads=1)
    params = attn.init(key, inputs)
    outputs = attn.apply(params, inputs)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, height, width, channels)


def test_down_block():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    out_channels = 128
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    db = DownBlock(features=out_channels, with_attention=True)
    params = db.init(key, inputs, times)
    outputs = db.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, height, width, out_channels)


def test_up_block():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    out_channels = 32
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    ub = UpBlock(features=out_channels, with_attention=True)
    params = ub.init(key, inputs, times)
    outputs = ub.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, height, width, out_channels)


def test_middle_block():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    out_channels = 32
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.uniform(key, shape=(batch_size, channels), minval=0, maxval=1, dtype=jnp.float32)

    mb = MiddleBlock(features=out_channels)
    params = mb.init(key, inputs, times)
    outputs = mb.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, height, width, out_channels)


def test_down_sample():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)

    ds = DownSample()
    params = ds.init(key, inputs)
    outputs = ds.apply(params, inputs)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, height / 2, width / 2, channels)


def test_up_sample():
    batch_size = 128
    height, width = 32, 48
    channels = 64
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)

    us = UpSample()
    params = us.init(key, inputs)
    outputs = us.apply(params, inputs)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, 2 * height, 2 * width, channels)


def test_unet():
    batch_size = 128
    height, width = 32, 48
    channels = 3
    initial_channels = 64
    num_steps = 1000
    key = random.key(0)
    inputs = jnp.ones((batch_size, height, width, channels), jnp.float32)
    times = random.randint(key, shape=(batch_size,), minval=0, maxval=num_steps)

    unet = UNet(
        num_channels=initial_channels,
        channel_multipliers=(1, 2, 2, 4),
        use_attention=(False, False, True, True),
        num_blocks=2,
    )
    params = unet.init(key, inputs, times)
    outputs = unet.apply(params, inputs, times)
    assert isinstance(outputs, Array)
    assert outputs.shape == (batch_size, height, width, channels)
