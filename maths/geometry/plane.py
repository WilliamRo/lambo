import numpy as np
from scipy.ndimage import median_filter


def fit_plane_adaptively(
    x: np.ndarray, n_points=200, low=1, high=(10, 25, 40, 60), max_rounds=3,
    alpha=0.2, filter_size: int = 0):
  """Adaptively fit plane based on the criteria below
           score = b - alpha * r
      in which b is global background score and r is the local fitting residual
  """
  # Sanity check
  if not isinstance(high, (tuple, list)):
    assert isinstance(high, int) and high > low
    high = high,

  # Initialize some variables
  bg = 0
  best_score, best_p, best_r, best_b = -np.infty, None, None, None

  # Run at most k iterations
  for k in range(max_rounds):
    x_base = x if filter_size == 0  else median_filter(x, filter_size)

    # Go over all `high` values, select the one with highest score
    mask_base = x_base - bg
    bg, p, r, b = sorted(
      [fit_plane(x_base, n_points, low, h, mask_base) for h in high],
      key=lambda z: z[3] - alpha * z[2], reverse=True)[0]
    score = b - alpha * r

    # Take down if necessary
    if score > best_score:
      bg, best_score, best_p, best_r, best_b = bg, score, p, r, b
    else:
      break

  # Return best results
  assert best_p is not None
  return bg, best_p, best_r, best_b


def fit_plane(x: np.ndarray, n_points=100, low=1, high=30, mask_base=None):
  """Given an input image of shape (H, W), fit a plane
       z = a*x + b*y + c
    that properly fit the background of this image.

    Assume background points are {x_i, y_i, z_i}_{i=1}^N, the problems becomes
       min_p ||A \cdot p - z||_2
    with an analytical solution
      p = [a, b, c]^T = (A^T \cdot A)^{-1} \cdot A^T \cdot z
    where A[i] = [x_i, y_i, 1].
  """
  # Get background mask
  if mask_base is None: mask_base = x
  mask = get_bg_mask(mask_base, low=low, high=high)

  # Points in mask are too many, we only use at most `n_points` pixels to fit
  #   the plane
  coord = np.argwhere(mask)
  step = max(1, len(coord) // n_points)
  coord = coord[::step]

  # Fit a plane
  A = np.vstack([[i, j, 1.0] for i, j in coord])
  z = [x[i, j] for i, j in coord]
  p, r = np.linalg.lstsq(A, z, rcond=None)[:2]

  # Calculate background and flatness
  H, W = x.shape
  X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
  bg = p[0] * X + p[1] * Y + p[2]

  # Calculate local fitting score
  r = 1000 * r[0] / n_points

  # Calculate global fitting score
  b = np.mean(1. / (1. + np.square(x - bg))) * 100

  # Return
  return bg, p, r, b


def get_bg_mask(x: np.ndarray, low=0.5, high=20):
  bg_min = np.percentile(x, low)
  bg_max = np.percentile(x, high)
  return (x > bg_min) * (x < bg_max)


def analyze_bg_extraction(x: np.ndarray, **kwargs):
  from matplotlib.gridspec import GridSpec
  from matplotlib import cm
  from typing import Optional
  from lambo.gui.pyplt.events import bind_quick_close

  import matplotlib.pyplot as plt

  # Define some methods for plot
  def _plot_surface(
      ax: plt.Axes, array: np.ndarray, title: Optional[str] = None):
    H, W = array.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    ax.plot_surface(X, Y, array, cmap=cm.coolwarm)
    if title: ax.set_title(title)

  def _plot_hist(array: np.ndarray, title: str):
    plt.hist(np.ravel(array), bins=20, density=True)
    plt.title('{}, Range: [{:.2f}, {:.2f}]'.format(
      title, np.min(array), np.max(array)))

  def _plot_bg(ax: plt.Axes, m: np.ndarray, title):
    coordinates = np.argwhere(m)
    coordinates = coordinates[::100]
    X = [c[0] for c in coordinates]
    Y = [c[1] for c in coordinates]
    Z = [x[c[0], c[1]] for c in coordinates]
    # ax.plot3D(X, Y, Z, '.', linewidth='0.01')
    ax.scatter(X, Y, Z, c=Z)
    plt.title(title)

  # Initialize figure
  fig = plt.figure(figsize=(12, 8))
  assert isinstance(fig, plt.Figure)
  spec = GridSpec(nrows=2, ncols=2, height_ratios=[3, 1])

  # Left part
  ax = fig.add_subplot(spec[0], projection='3d')
  assert isinstance(ax, plt.Axes)
  _plot_surface(ax, x, 'Unwrapped Phase')
  fig.add_subplot(spec[2])
  _plot_hist(x, 'Unwrapped Phase')

  # Plot right part
  x_lim, y_lim, z_lim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
  ax = fig.add_subplot(spec[1], projection='3d')
  mask = get_bg_mask(x)
  bg = x * mask
  # Plot bg
  _plot_bg(ax, bg, 'Background')
  ax.set_xlim(x_lim)
  ax.set_ylim(y_lim)
  ax.set_zlim(z_lim)

  fig.add_subplot(spec[3])
  _plot_hist(bg, 'Background')
  assert isinstance(ax, plt.Axes)

  # Finalize
  bind_quick_close()
  plt.tight_layout()
  plt.show()


def fit_plane_v1_deprecated(img: np.ndarray, radius=0.05):
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