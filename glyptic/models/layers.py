import math
from typing import Sequence

import flax.linen as nn
from jax import Array
import jax
import jax.numpy as jnp


class SinusoidalEmbedding(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, times: Array) -> Array:
        # Half length of the desired embedding length
        L = self.num_channels // 8
        # Maximum scaling factor for frequencies
        emb = math.log(10000) / (L - 1)
        emb = jnp.exp(jnp.arange(L) * -emb)
        emb = times[:, None] * emb[None, :]
        emb = jnp.concat((jnp.sin(emb), jnp.cos(emb)), axis=1)

        # Apply MLP transformation
        emb = nn.Dense(self.num_channels)(emb)
        emb = nn.swish(emb)
        emb = nn.Dense(self.num_channels)(emb)
        return emb


class ResidualBlock(nn.Module):
    filters: int
    dropout_rate: float
    train: bool
    num_groups: int = 32

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        residual = x
        out = nn.GroupNorm(num_groups=self.num_groups)(x)
        out = nn.swish(out)
        out = nn.Conv(self.filters, kernel_size=(3, 3), padding=(1, 1))(out)

        times = nn.swish(times)
        out += nn.Dense(self.filters)(times)[:, None, None, :]

        out = nn.GroupNorm(num_groups=self.num_groups)(out)
        out = nn.swish(out)
        out = nn.Dropout(self.dropout_rate, deterministic=True)(out)
        # out = nn.Dropout(self.dropout_rate, deterministic=not self.train)(out)
        out = nn.Conv(self.filters, kernel_size=(3, 3), padding=(1, 1))(out)

        if residual.shape != out.shape:
            residual = nn.Conv(self.filters, kernel_size=(1, 1))(residual)
        return out + residual


# TODO: Test with flax built-in MultiHeadDotProductAttention
class AttentionBlock(nn.Module):
    num_heads: int = 1
    d_k: int | None = None
    num_groups: int = 32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # X is BHWC
        input_channels: int = x.shape[-1]
        d_k = self.d_k if self.d_k else input_channels

        batch_size, height, width, num_channels = x.shape

        x = x.reshape(batch_size, -1, num_channels)

        qkv = nn.Dense(self.num_heads * d_k * 3)(x)
        qkv = qkv.reshape(batch_size, -1, self.num_heads, 3 * d_k)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        attention = jnp.einsum("bihd,bjhd->bijh", q, k) * (d_k**-0.5)
        attention = nn.softmax(attention, axis=2)

        out = jnp.einsum("bijh,bjhd->bihd", attention, v)
        out = out.reshape(batch_size, -1, self.num_heads * d_k)
        out = nn.Dense(input_channels)(out)
        out += x
        out = out.reshape(batch_size, height, width, num_channels)
        return out


class DownBlock(nn.Module):
    features: int
    dropout_rate: float
    train: bool
    with_attention: bool = False

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        out = ResidualBlock(
            filters=self.features, dropout_rate=self.dropout_rate, train=self.train
        )(x, times)
        if self.with_attention:
            # TODO: Determine if it would be better to have groupnorm and skip here also?
            out = AttentionBlock()(out)

        return out


class UpBlock(nn.Module):
    features: int
    dropout_rate: float
    train: bool
    with_attention: bool = False

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        out = ResidualBlock(
            filters=self.features, dropout_rate=self.dropout_rate, train=self.train
        )(x, times)
        if self.with_attention:
            out = AttentionBlock()(out)
        return out


class MiddleBlock(nn.Module):
    features: int
    dropout_rate: float
    train: bool

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        out = ResidualBlock(
            filters=self.features, dropout_rate=self.dropout_rate, train=self.train
        )(x, times)
        # out = AttentionBlock()(out)
        # TODO: MISTAKE original called (x, times) instead of (out, times)
        out = ResidualBlock(
            filters=self.features, dropout_rate=self.dropout_rate, train=self.train
        )(out, times)
        return out


class DownSample(nn.Module):
    @nn.compact
    def __call__(self, x: Array, *_) -> Array:
        # Keeping number of channels the same
        # features = x.shape[-1]
        # out = nn.Conv(features, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1))(x)

        out = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return out


# Scale up feature map 2x
class UpSample(nn.Module):
    @nn.compact
    def __call__(self, x: Array, *_) -> Array:
        # Keeping number of channels the same
        # features = x.shape[-1]
        # TODO: Might need to change as not equivalent to torch ConvTranspose2D
        # https://github.com/google/flax/issues/1872
        # out = nn.ConvTranspose(features, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)

        B, H, W, C = x.shape
        out = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method="bilinear")
        out = nn.Conv(C, kernel_size=(3, 3), padding=(1, 1))(out)
        return out


class UNet(nn.Module):
    train: bool
    num_channels: int = 64
    channel_multipliers: Sequence[int] = (1, 2, 2, 4)
    use_attention: Sequence[bool] = (False, False, True, True)
    num_blocks: int = 2
    dropout_rate: float = 0.1

    def setup(self):
        num_resolutions = len(self.channel_multipliers)
        self.image_projection = nn.Conv(self.num_channels, kernel_size=(3, 3), padding=(1, 1))

        self.time_embedding = SinusoidalEmbedding(self.num_channels * 4)

        down_blocks = []

        out_channels = self.num_channels

        for i in range(num_resolutions):
            out_channels = out_channels * self.channel_multipliers[i]

            for _ in range(self.num_blocks):
                down_blocks.append(
                    DownBlock(
                        out_channels,
                        dropout_rate=self.dropout_rate,
                        train=self.train,
                        with_attention=self.use_attention[i],
                    )
                )

            if i < num_resolutions - 1:
                down_blocks.append(DownSample())

        self.down_blocks = down_blocks

        self.middle_block = MiddleBlock(
            out_channels, dropout_rate=self.dropout_rate, train=self.train
        )

        up_blocks = []

        for i in reversed(range(num_resolutions)):
            for _ in range(self.num_blocks):
                up_blocks.append(
                    UpBlock(
                        out_channels,
                        dropout_rate=self.dropout_rate,
                        train=self.train,
                        with_attention=self.use_attention[i],
                    )
                )

            out_channels = out_channels // self.channel_multipliers[i]
            up_blocks.append(
                UpBlock(
                    out_channels,
                    dropout_rate=self.dropout_rate,
                    train=self.train,
                    with_attention=self.use_attention[i],
                )
            )

            if i > 0:
                up_blocks.append(UpSample())

        self.up_blocks = up_blocks

        self.group_norm = nn.GroupNorm(8)
        # image_channels = 3 (RGB)
        self.feature_aggregation = nn.Conv(3, kernel_size=(3, 3), padding=(1, 1))

    def __call__(self, x: Array, times: Array) -> Array:
        x = self.image_projection(x)
        times = self.time_embedding(times)  # batch_size, num_channels

        hidden_states = [x]

        out = x
        for block in self.down_blocks:
            out = block(out, times)
            hidden_states.append(out)

        out = self.middle_block(out, times)

        for block in self.up_blocks:
            if isinstance(block, UpSample):
                out = block(out, times)
            else:
                skip = hidden_states.pop()
                out = jnp.concat((out, skip), axis=3)
                out = block(out, times)

        out = self.group_norm(out)
        out = nn.swish(out)
        out = self.feature_aggregation(out)

        return out
