import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import einops
from jax import lax
from typing import Optional, List


class GroupNorm(nn.Module):

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x = nn.GroupNorm(n_groups=c)(x)
        return x


class DropPath(nn.Module):
    survival_prob: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        if self.survival_prob == 1.:
            return inputs
        elif self.survival_prob == 0.:
            return jnp.zeros_like(inputs)

        if deterministic:
            return inputs
        else:
            rng = self.make_rng('droppath')
            broadcast_shape = [inputs[0].shape[0]] + [1 for _ in range(len(inputs[0].shape) - 1)]
            epsilon = jax.random.bernoulli(key=rng,
                                           p=self.survival_prob,
                                           shape=broadcast_shape
                                           )
            return inputs / self.survival_prob * epsilon


class Pooling(nn.Module):
    pool_window: int = 3
    strides: int = 1

    @nn.compact
    def __call__(self, x):
        x_sum = lax.reduce_window(
            x, 0., lax.add, (1, self.pool_window, self.pool_window, 1),
            (1, self.strides, self.strides, 1), 'SAME'
        )
        div_term = lax.reduce_window(
            jnp.ones_like(x), 0., lax.add, (1, self.pool_window, self.pool_window, 1),
            (1, self.strides, self.strides, 1), 'SAME'
        )
        x_pooled = x_sum / div_term
        return x_pooled - x


class ChannelMLP(nn.Module):
    r: int
    act: str = 'gelu'

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        x = nn.Dense(c * self.r)
        if self.act == 'relu':
            x = nn.relu(x)
        elif self.act == 'gelu':
            x = nn.gelu(x)
        elif self.act == 'hardswish':
            x = nn.hard_swish(x)
        else:
            raise NotImplementedError('Activation error')
        x = nn.Dense(c)(x)
        return x

