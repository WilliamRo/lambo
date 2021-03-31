import numpy as np


def fit_plane(img: np.ndarray, radius=0.05):
  """Fit a plane to be subtracted from the given img so that img can be
  flattened.
                j | W
      ---------------------
     | 0  | r           1 |
     |____| r             |
     |                    |
     |                    |  i | H
     |                    |
     |                    |
     | 2                3 |
     ---------------------

  :param img: img that is not flat
  :param radius: radius of area used to calculate median value
  """
  # Get image shape
  assert len(img.shape) == 2
  H, W = img.shape
  ri, rj = [max(1, int(s * radius)) for s in img.shape]

  # Get the representative values of 4 sub-corners
  sub_corners = [np.median(img[i-ri:i+ri+1, j-rj:j+rj+1])
                 for i in (ri, H - ri - 1) for j in (rj, W - rj - 1)]

  # Calculate real conner values
  rs = [None, (W - 2 * rj - 1) * rj, (H - 2 * ri - 1) * ri]
  get_delta = lambda i1, i2: (sub_corners[i2] - sub_corners[i1]) / rs[i2 - i1]
  d_ud_l, d_ud_r = get_delta(0, 2), get_delta(1, 3)
  d_lr_u, d_lr_d = get_delta(0, 1), get_delta(2, 3)
  corners = [sub_corners[0] - d_lr_u - d_ud_l,
             sub_corners[1] + d_lr_u - d_ud_r,
             sub_corners[2] - d_lr_d + d_ud_l,
             sub_corners[3] + d_lr_d + d_ud_r]

  # Generate horizontal lines
  horizontal_u = np.linspace(corners[0], corners[1], W)
  horizontal_d = np.linspace(corners[2], corners[3], W)

  # Generate plane and return
  plane = np.linspace(horizontal_u, horizontal_d, H)
  return plane


if __name__ == '__main__':
  img = np.linspace(
    np.linspace(1, 15, 15), np.linspace(10, 24, 15), 10).transpose()
  print('img.shape = {}, img =\n{}'.format(img.shape, img))
  plane = fit_plane(img)
  assert np.linalg.norm(img - plane) == 0