import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import einops
from jax import lax
from typing import Optional, List
from layers import *


class PFBlock(nn.Module):
    survival_prob: float
    depth: int
    pool_size: int = 3
    pool_stride: int = 1

    @nn.compact
    def __call__(self, x, deterministic):
        b, h, w, c = x.shape
        scale_init = .1 if self.depth < 18 else 1e-5 if self.depth < 24 else 1e-6
        token_scale = self.param('token_scale',
                                 nn.initializers.constant(scale_init),
                                 (1, 1, 1, c)
                                 )
        channel_scale = self.param('channel_scale',
                                   nn.initializers.constant(scale_init),
                                   (1, 1, 1, c)
                                   )

        x_res = GroupNorm()(x)
        x_res = Pooling()(x_res)
        x_res = x_res * token_scale
        x = x + DropPath(self.survival_prob)(x_res, deterministic=deterministic)

        x_res = GroupNorm()(x)
        x_res = ChannelMLP()(x_res)
        x_res = x_res * channel_scale
        x = x + DropPath(self.survival_prob)(x_res, deterministic=deterministic)

        return x


class PoolFormer(nn.Module):
    # S36
    n_labels: int = 1000
    stochastic_depth_rate: float = .1
    embedding_kernel: List[int] = [7, 3, 3, 3]
    embedding_strides: List[int] = [4, 2, 2, 2]
    embedding_filters: List[int] = [64, 128, 320, 512]
    n_blocks: List[int] = [6, 6, 18, 6]

    @nn.compact
    def __call__(self, x, deterministic=True):
        survival_prob = 1. - jnp.linspace(0., self.stochastic_depth_rate, sum(self.n_blocks))

        # Feature Extractor
        for i in range(len(self.n_blocks)):
            x = nn.Conv(self.embedding_filters[i],
                        (self.embedding_kernel[i], self.embedding_kernel[i]),
                        (self.embedding_strides[i], self.embedding_strides[i])
                        )(x)
            depth = sum(self.n_blocks[:i])
            for b in range(self.n_blocks[i]):
                x = PFBlock(survival_prob[depth + b],
                            depth + b
                            )(x)

        # Classification Head
        x = jnp.mean(x, axis=(1, 2))
        x = GroupNorm()(x)
        x = nn.Dense(self.n_labels,
                     kernel_init=nn.initializers.zeros
                     )(x)
        y_hat = nn.softmax(x)
        return y_hat
