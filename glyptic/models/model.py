from jax import jit, random, vmap
import jax.numpy as jnp


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(z):
    return jnp.maximum(0, z)


# TODO: Revise predict func, esp. output values (currently probs)
@jit
def forward(params, x):
    a = x
    for w, b in params[:-1]:
        z = jnp.dot(w, a) + b
        a = relu(z)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, a) + final_b
    # TODO: Figure out activation func.
    return logits


@jit
def batch_forward(params, batched_x):
    return vmap(forward, in_axes=(None, 0))(params, batched_x)
