import numpy as np
import matplotlib.pyplot as plt
import time

from lambo.gui.vinci.vinci import DaVinci
from scipy.ndimage import median_filter

from lambo.maths.geometry.plane import get_bg_mask


class PlaneDaVinci(DaVinci):

  def __init__(self, x: np.ndarray, size: int = 7):
    # Call parent's constructor
    super(PlaneDaVinci, self).__init__('Plane Fitting', height=size, width=size)

    # Set object
    self.objects.append(x)
    # View input image
    self.add_plotter(lambda ax: self.imshow(x, ax, title='Unwrapped Phase'))
    self.add_plotter(
      lambda im, ax3d: self.plot3d(im, ax3d, title='Unwrapped Phase'))


  def plot_anchors(self, im: np.ndarray, mask: np.ndarray, ax3d: plt.Axes,
                   points: int = 100):
    coordinates = np.argwhere(mask)
    step = len(coordinates) // points
    coordinates = coordinates[::step]
    X = [c[0] for c in coordinates]
    Y = [c[1] for c in coordinates]
    Z = [im[c[0], c[1]] for c in coordinates]
    self.scatter(X, Y, Z, ax3d)


  def show_im_and_plane(
      self, im: np.ndarray, plane: np.ndarray, ax3d: plt.Axes, title=None):
    self.plot3d(im, ax3d, alpha=0.3)
    self.plot_wireframe(plane, ax3d)
    if title is not None: ax3d.set_title(title)


  def show_im_and_anchor(self, im, mask, ax3d, points=50, title=None):
    self.plot3d(im, ax3d, alpha=0.3)
    self.plot_anchors(im, mask, ax3d, points=points)
    if title is not None: ax3d.set_title(title)


  def fit_and_show(self, n_points=200, low=1, high=25, filter_size=0,
                   max_rounds=3, alpha=0.2, show_mask_base=False):

    global_tic = time.time()
    print('>> Begin fitting ...')

    x = self.objects[0]
    z_min, z_max = np.min(x), np.max(x)

    # Initialize some variables
    bg = 0
    best_score, best_bg = -np.infty, None

    # Define fit_plane function
    def fit_plane(x, x_base, mask_base, k):

      # (*) Visualize mask_base if required
      if show_mask_base and filter_size > 0:
        self.add_plotter(lambda ax3d: self.plot3d(
          mask_base, ax3d, title='[Round {}] Mask base'.format(k + 1)))

      # Get background mask
      bg_min = np.percentile(mask_base, low)
      bg_max = np.percentile(mask_base, high)
      mask = (mask_base > bg_min) * (mask_base < bg_max)

      # Get background coordinates given mask
      coord = np.argwhere(mask)
      step = max(1, len(coord) // n_points)
      coord = coord[::step]

      # (*) Visualize anchors
      self.add_plotter(lambda ax3d: self.show_im_and_anchor(
        x, mask, ax3d, points=30,
        title='[Round {}] Anchors'.format(k + 1)))

      # Fit a plane
      tic = time.time()
      A = np.vstack([[i, j, 1.0] for i, j in coord])
      z = [x_base[i, j] for i, j in coord]
      p, r = np.linalg.lstsq(A, z, rcond=None)[:2]
      print('++ Time elapsed for fitting: {:.2f}'.format(
        time.time() - tic))

      # Calculate background and flatness
      H, W = x.shape
      X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
      bg = p[0] * X + p[1] * Y + p[2]

      # Calculate local fitting score
      r = 1000 * r[0] / n_points
      # Calculate global fitting score
      b = np.mean(1. / (1. + np.square(x - bg))) * 100
      # Calculate score
      score = b - alpha * r

      # Display title
      title = '[Round {}] Score = {:.2f}'.format(k + 1, score)
      print(title)

      # (*) Visualize plane
      self.add_plotter(
        lambda ax3d: self.show_im_and_plane(x, bg, ax3d, title=title))

      # Return
      return bg, score

    # Run at most k iterations
    for k in range(max_rounds):

      # Get mask
      if filter_size > 0:
        print('>> Applying median filter (size = {})'.format(filter_size))
        tic = time.time()
        x_base = median_filter(x, size=filter_size)
        print('++ Time elapsed for median_filter: {:.2f}'.format(
          time.time() - tic))
      else: x_base = x

      # Go over all `high` values, select the one with highest score
      mask_base = x_base - bg

      # Fit plane for round k + 1
      bg, score = fit_plane(x, x_base, mask_base, k)

      # Record
      if score > best_score: best_bg, best_score = bg, score
      else: break

    # Show global time elapsed
    print('++ Total time elapsed: {:.2f} secs'.format(time.time() - global_tic))

    # Show flattened image
    flattened = x - best_bg
    title = 'Flattened Phase, Score = {:.2f}'.format(best_score)
    self.add_plotter(lambda ax3d: self.plot3d(flattened, ax3d, title=title))
    self.add_plotter(lambda ax: self.imshow(flattened, ax, title=title))

    # Find
    z_min = np.minimum(z_min, np.min(flattened))
    z_max = np.maximum(z_max, np.max(flattened))

    # Show
    self.z_lim = (z_min - 1, z_max + 1)
    self.show()


if __name__ == '__main__':
  print('>> Importing tframe utilities ...')
  from pr.pr_agent import PRAgent

  data_dir = r'E:\lambai\01-PR\data'

  # (3, 6/7), (1, 6)* hard to flat
  # (1, 1) ill
  # (3, 2), (1, 2)*: steep

  # (5, 4/5/8/11/12/31): bead on edge
  # (5, 42): nearly full of beads
  # (5, 40): full of beads

  # trial_id, sample_id = 3, 6
  trial_id, sample_id = 5, 40
  pattern = None
  # pattern = '70'

  print('>> Reading interferogram ...')
  ig = PRAgent.read_interferogram(
    data_dir, trial_id, sample_id, pattern=pattern)

  da = PlaneDaVinci(ig.unwrapped_phase)
  da.keep_3D_view_angle = True
  da.fit_and_show(
    show_mask_base=True,
    n_points=200,
    low=1,
    high=50,
    max_rounds=1,
    filter_size=0,
    alpha=0.2)