from typing import Union
import numpy as np


def rotate_coord(coord: Union[tuple, list, np.ndarray], angle: float):
  def _convert(x, form='array') -> np.ndarray:
    assert form in ('array', 'matrix')
    if isinstance(x, (tuple, list)): x = np.array(x)
    assert isinstance(x, np.ndarray) and x.size == 2
    if form == 'array': return x.flatten()
    else: return x.reshape(2, 1)

  if angle == 0: return _convert(coord)

  # Rotate coordinate
  theta = 1. * angle / 180. * np.pi

  # Generate rotation matrix
  sin, cos = np.sin(theta), np.cos(theta)
  M = np.array([[cos, -sin], [sin, cos]])
  coord = np.matmul(M, _convert(coord, 'matrix'))

  # Return result
  return coord.flatten()

