import flax.linen as nn


class ResidualBlock(nn.Module):
    filters: int
    num_groups: int = 32

    @nn.compact
    def __call__(self, x, times):
        # TODO: Check: order of norm, act, conv; whether to norm on residual
        # TODO: Add dropout
        residual = x
        out = nn.GroupNorm(num_groups=self.num_groups)(x)
        out = nn.swish()(out)
        out = nn.Conv(self.filters, kernel_size=(3, 3), padding=(1, 1))

        times = nn.swish()(times)
        # TODO: Some sort of reshape, in PyTorch [:, :, None, None]
        # TODO: Might not need embeddings added to each res block??
        out += nn.Dense(times.shape[1])(times)

        out = nn.GroupNorm(num_groups=self.num_grousp)(out)
        out = nn.swish()(out)
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
