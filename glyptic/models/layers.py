import jax.numpy as jnp
import flax.linen as nn
import math


class SinusoidalEmbedding(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, times):
        # Half length of the desired embedding length
        L = self.num_channels // 8
        # Maximum scaling factor for frequencies
        emb = math.log(10000) / (L - 1)
        emb = jnp.exp(jnp.arange(L) * -emb)
        emb = times[:, None] * emb[None, :]
        emb = jnp.concat((emb.sin(), emb.cos()), axis=1)

        # Apply MLP transformation
        emb = nn.Dense(self.num_channels)(emb)
        emb = nn.swish()(emb)
        emb = nn.Dense(self.num_channels)(emb)
        return emb


class ResidualBlock(nn.Module):
    filters: int
    num_groups: int = 32

    @nn.compact
    def __call__(self, x, times):
        # TODO: Check: order of norm, act, conv; whether to norm on residual
        # TODO: Add dropout
        residual = x
        out = nn.GroupNorm(num_groups=self.num_groups)(x)
        out = nn.swish(out)
        out = nn.Conv(self.filters, kernel_size=(3, 3), padding=(1, 1))(out)

        times = nn.swish(times)
        # Adds two dimensions for H, W.
        # TODO: Might not need time embeddings added to each res block??
        out += nn.Dense(self.filters)(times)[:, None, None, :]

        out = nn.GroupNorm(num_groups=self.num_groups)(out)
        out = nn.swish(out)
        out = nn.Conv(self.filters, kernel_size=(3, 3), padding=(1, 1))(out)

        if residual.shape != out.shape:
            residual = nn.Conv(self.filters, kernel_size=(1, 1))(residual)
        return out + residual


class DownBlock(nn.Module):
    # TODO: Call something else
    units: int

    @nn.compact
    def __call__(self, x, times):
        out = ResidualBlock(filters=32, num_groups=32)(x, times)
        return x
