import numpy as np


def random_window(window_shape, full_shape, sigma=None):
  """Randomly samples a sub-window with given shape (hxw) in a 2D array A
  (with shape HxW). The location of upper-left corner (i, j) will be returned.
  Thus the sampled window is A[i:i+h, j:j+w]
  """
  # TODO: currently normal distribution for sampling is not supported
  assert sigma is None
  h, w = window_shape
  H, W = full_shape
  return [np.random.randint(0, high + 1) for high in (H - h, W - w)]


if __name__ == '__main__':
  for _ in range(10):
    print(random_window([6, 8], [20, 40]))