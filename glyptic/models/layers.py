import jax.numpy as jnp
import flax.linen as nn
import math
from jax import Array


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
    num_groups: int = 32

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        # TODO: Check: order of norm, act, conv; whether to norm on residual
        # TODO: Add dropout
        residual = x
        out = nn.GroupNorm(num_groups=self.num_groups)(x)
        out = nn.swish(out)
        out = nn.Conv(self.filters, kernel_size=(3, 3), padding=(1, 1))(out)

        times = nn.swish(times)
        # Adds two dimensions for H, W of time array.
        # TODO: Might not need time embeddings added to each res block??
        out += nn.Dense(self.filters)(times)[:, None, None, :]

        out = nn.GroupNorm(num_groups=self.num_groups)(out)
        out = nn.swish(out)
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
    with_attention: bool = False

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        out = ResidualBlock(filters=self.features)(x, times)
        if self.with_attention:
            out = AttentionBlock()(out)

        return out


class UpBlock(nn.Module):
    features: int
    with_attention: bool = False

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        out = ResidualBlock(filters=self.features)(x, times)
        if self.with_attention:
            out = AttentionBlock()(out)
        return out


class MiddleBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: Array, times: Array) -> Array:
        out = ResidualBlock(filters=self.features)(x, times)
        out = AttentionBlock()(out)
        out = ResidualBlock(filters=self.features)(x, times)
        return out


class DownSample(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        out = nn.Conv(self.features, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1))(x)
        return out


# Scale up feature map 2x
class UpSample(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # TODO: Might need to change as not equivalent to torch ConvTranspose2D
        out = nn.ConvTranspose(self.features, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
        return out
