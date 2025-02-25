from flax import struct
import jax.numpy as jnp


class RollingAverage(struct.PyTreeNode):
    """
    Identical to the example in style, but using JAX arrays and flax.struct.
    Keeps track of a ring buffer of values, returns rolling mean.
    """

    size: int
    last_element: int
    mat: jnp.ndarray

    def update(self, new_value: float):
        mat = self.mat.at[self.last_element].set(new_value)
        last_element = (self.last_element + 1) % mat.shape[0]
        size = jnp.where(self.size != mat.shape[0], self.size + 1, self.size)
        curr_value = mat.sum() / size
        return curr_value, self.replace(size=size, last_element=last_element, mat=mat)

    @classmethod
    def create(cls, *, size: int):
        return cls(
            size=0,
            last_element=0,
            mat=jnp.zeros(size, dtype=jnp.float32),
        )
