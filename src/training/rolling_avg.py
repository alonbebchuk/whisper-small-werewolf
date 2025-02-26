import jax.numpy as jnp

from flax.struct import PyTreeNode


class RollingAverage(PyTreeNode):
    size: int
    last_element: int
    mat: jnp.ndarray

    def update(self, new_value):
        mat = self.mat.at[self.last_element].set(new_value)
        last_element = (self.last_element + 1) % mat.shape[0]
        size = jnp.where(self.size != mat.shape[0], self.size + 1, self.size)

        curr_value = mat.sum() / size
        new_value = self.replace(size=size, last_element=last_element, mat=mat)
        return curr_value, new_value

    @classmethod
    def create(cls, *, size: int):
        rolling_average = cls(size=0, last_element=0, mat=jnp.zeros(size, dtype=jnp.float32))
        return rolling_average
