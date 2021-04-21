import numpy as np
import matplotlib.pyplot as plt
import roma

from typing import Optional
from lambo.data_obj.image import DigitalImage
from lambo.gui.pyplt import imshow
from lambo.gui.pyplt.events import bind_quick_close
from lambo.maths.geometry.plane import fit_plane

from skimage.restoration import unwrap_phase


class Interferogram(DigitalImage):

  PEAK_BOUND_PCT = 0.6

  def __init__(self, img, bg=None, radius=None, lambda_0=None, delta_n=None,
               **kwargs):
    # Call parent's initializer
    super(Interferogram, self).__init__(img, **kwargs)
    # Attributes
    self.radius = roma.check_type(radius, int, nullable=True)
    self.lambda_0 = roma.check_type(lambda_0, float, nullable=True)
    self.delta_n = roma.check_type(delta_n, float, nullable=True)
    self._background: Optional[Interferogram] = None
    if bg is not None: self.set_background(bg)

    self.sample_token = None
    self.setup_token = None

  # region: Properties

  @property
  def log_Sc(self) -> np.ndarray:
    return np.log(self.Sc + 1)

  @property
  def bg_array(self) -> np.ndarray:
    return self._background.img

  @property
  def peak_index(self) -> tuple:
    def _find_peak():
      bound = int(self.PEAK_BOUND_PCT * self.Sc.shape[0])
      region = self.Sc[bound:, :]
      index = np.unravel_index(np.argmax(region), region.shape)
      return (index[0] + bound, index[1])
    return self.get_from_pocket('index_of_+1_point', initializer=_find_peak)

  @peak_index.setter
  def peak_index(self, val):
    self.put_into_pocket(
      'index_of_+1_point', roma.check_type(val, inner_type=int))

  @property
  def mask(self) -> np.ndarray:
    # TODO: consider apodized band-pass filter
    def _get_mask():
      H, W = self.Sc.shape
      ci, cj = self.peak_index
      X, Y = np.ogrid[:H, :W]
      return np.sqrt((X - ci)**2 + (Y - cj)**2) <= self.radius
    return self.get_from_pocket('mask_of_+1_point', initializer=_get_mask)

  @property
  def homing_signal(self) -> np.ndarray:
    def _homing_signal():
      masked = self.Fc * self.mask
      CI, CJ = [s // 2 for s in self.Fc.shape]
      pi, pj = self.peak_index
      return np.roll(masked, shift=(CI - pi, CJ - pj), axis=(0, 1))
    return self.get_from_pocket('homing_masked_Fc', initializer=_homing_signal)

  @property
  def extracted_image(self) -> np.ndarray:
    return self.get_from_pocket(
      'extracted_image',
      initializer=lambda: np.fft.ifft2(np.fft.ifftshift(self.homing_signal)))

  @property
  def delta_angle(self) -> np.ndarray:
    """Phase information after subtracting background"""
    assert all([isinstance(self._background, Interferogram),
                self.size == self._background.size])
    return self.get_from_pocket('retrieved_phase', initializer=lambda: np.angle(
      self.extracted_image / self._background.extracted_image))

  @property
  def unwrapped_phase(self) -> np.ndarray:
    """Result after performing phase unwrapping"""
    return self.get_from_pocket(
      'unwrapped_phase', initializer=lambda: unwrap_phase(self.delta_angle))

  @property
  def bg_plane_info(self) -> np.ndarray:
    _fit_plane = lambda: fit_plane(self.unwrapped_phase.copy())
    return self.get_from_pocket('bg_plane_info', initializer=_fit_plane)

  @property
  def flattened_phase(self) -> np.ndarray:
    def _flattened_phase():
      phase = self.unwrapped_phase.copy()
      bg_plane, _ = self.bg_plane_info
      flattened = phase - bg_plane
      flattened = np.maximum(flattened, 0)
      return flattened
    return self.get_from_pocket('flattened_phase', initializer=_flattened_phase)

  @property
  def sample_height(self):
    def _height():
      lambda_0 = roma.check_type(self.lambda_0, float)
      delta_n = roma.check_type(self.delta_n, float)
      return self.flattened_phase * lambda_0 / (2 * np.pi * delta_n)
    return self.get_from_pocket('sample_height', initializer=lambda: _height)

  # endregion: Properties

  # region: Public Methods

  def rotate(self, k):
    img = np.rot90(self.img, k)
    bg = np.rot90(self.bg_array, k)
    ig = Interferogram(img, bg, self.radius)
    return ig

  def soft_mask(self, alpha=0.1, mask_min=0.1):
    x = self.flattened_phase
    threshold = alpha * np.max(x)
    mask = x > threshold
    soft_mask = 1.0 * mask + (1.0 - mask) * x / threshold + mask_min
    return soft_mask

  def set_background(self, bg: np.ndarray):
    if self.radius is None:
      raise ValueError('!! radius must be specified before setting background')

    bg = Interferogram(bg, radius=self.radius)
    assert bg.size == self.size
    bg.peak_index = self.peak_index
    self._background = bg

  @classmethod
  def imread(cls, path, bg_path=None, radius=None, **kwargs):
    img = super().imread(path, return_array=True)
    bg = super().imread(bg_path, return_array=True) if bg_path else None
    return Interferogram(img, bg, radius)

  def imshow(self, interferogram=True, spectrum=False, extracted=False,
             angle=False, unwrapped=False, flattened=False, sample_mask=False,
             **kwargs):
    """Show this image
    :param show_fc: whether to show Fourier coefficients
    """
    imgs, titles = [], []
    def append_img(img, title):
      imgs.append(img)
      titles.append(title)

    # (0) interferogram
    if interferogram: append_img(self.img, 'Interferogram')
    # (1) spectrum
    if spectrum:
      im = np.log(self.Sc + 1)
      im = (im, lambda: plt.gca().add_artist(plt.Circle(
        list(reversed(self.peak_index)), self.radius, color='r', fill=False)))
      imgs.append(im)
      titles.append('Centered Spectrum (log)')
    # (2) extracted image
    if extracted: append_img(np.abs(self.extracted_image), 'Extracted')
    # (3) show retrieved phase
    if angle: append_img(self.delta_angle, 'Angle')
    # (4) Unwrapped phase image
    if unwrapped: append_img(self.unwrapped_phase, 'Unwrapped')
    # (5) Balanced phase image
    if flattened: append_img(self.flattened_phase, 'Flattened')
    # (6) Sample mask
    if sample_mask: append_img(self.soft_mask(), 'Sample Mask')

    # Show image using lambo.imshow
    imshow(*imgs, titles=titles, **kwargs)

  def phase_histogram(self, unwrapped=True, flattened=True):
    buffer = []
    def _plot_hist(array: np.ndarray, title: str, total: int):
      if total > 1: plt.subplot(1, total, len(buffer) + 1)
      plt.hist(np.ravel(array))
      plt.title('{}, Range: [{:.2f}, {:.2f}]'.format(
        title, np.min(array), np.max(array)))
      buffer.append(title)

    # Plot hist
    total = unwrapped + flattened
    size = 5
    plt.figure(figsize=(size * total, size))
    if unwrapped: _plot_hist(self.unwrapped_phase, 'Unwrapped Phase', total)
    if flattened: _plot_hist(self.flattened_phase, 'Flattened Phase', total)

    # Finalize
    plt.tight_layout()
    bind_quick_close()
    plt.show()

  def flatten_analysis(self, plot3d=False, show_plane=True):
    from matplotlib.gridspec import GridSpec
    from matplotlib import cm

    def _plot_img(ax: plt.Axes, array: np.ndarray, title=None):
      if plot3d:
        H, W = array.shape
        X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        ax.plot_surface(X, Y, array, cmap=cm.coolwarm)
      else: ax.imshow(array)
      if title: ax.set_title(title)

    def _plot_hist(array: np.ndarray, title: str):
      plt.hist(np.ravel(array), bins=20, density=True)
      plt.title('{}, Range: [{:.2f}, {:.2f}]'.format(
        title, np.min(array), np.max(array)))

    # Plot
    fig = plt.figure(figsize=(12, 8))
    spec = GridSpec(nrows=2, ncols=2, height_ratios=[3, 1])

    kwargs = {}
    left_array = self.unwrapped_phase
    if plot3d: kwargs['projection'] = '3d'
    ax = fig.add_subplot(spec[0], **kwargs)
    assert isinstance(ax, plt.Axes)
    _plot_img(ax, left_array, 'Unwrapped Phase')
    fig.add_subplot(spec[2])
    _plot_hist(left_array, 'Unwrapped Phase')

    # Plot plane if required
    if plot3d and show_plane:
      plane, r = self.bg_plane_info
      H, W = plane.shape
      X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
      ax.plot_wireframe(X, Y, plane, color='green')
      ax.set_title('Unwrapped Phase, r = {:.4f}'.format(r))

    # Plot right part
    z_lim = ax.get_zlim()

    right_array = self.flattened_phase
    ax = fig.add_subplot(spec[1], **kwargs)
    _plot_img(ax, right_array, 'Flattened Phase')
    fig.add_subplot(spec[3])
    _plot_hist(right_array, 'Flattened Phase')
    ax.set_zlim(z_lim)

    # Finalize
    bind_quick_close()
    plt.tight_layout()
    plt.show()

  # endregion: Public Methods

  # region: Analysis

  def analyze_windows(self, N, ignore_gap=0, max_cols=3, show_grid=0):
    """Divide img evenly into N by N parts and do phase retrieval individually.
    The difference between this result and the common non-dividing phase
    retrieval result will be shown.

    :param N: number of each dimension dividing into """
    h, w = [l // N for l in self.size]
    ground_truth =self.unwrapped_phase[:h*N, :w*N]
    assembled = np.zeros_like(ground_truth)
    for i, j in [(i, j) for i in range(N) for j in range(N)]:
      im = self.img[i*h:(i+1)*h, j*w:(j+1)*w]
      bg = self.bg_array[i*h:(i+1)*h, j*w:(j+1)*w]
      ig = Interferogram(im, bg, self.radius // N)
      assembled[i*h:(i+1)*h, j*w:(j+1)*w] = ig.unwrapped_phase

    # Compare
    delta = np.abs(ground_truth - assembled)

    # Ignore gap if required
    if ignore_gap:
      r = ignore_gap
      for i in range(1, N): delta[i * h - r:i * h + r, :] = 0
      for j in range(1, N): delta[:, j * w - r:j * w + r] = 0

    # Show grid if necessary
    if show_grid:
      r = show_grid
      v = np.min(assembled)
      for i in range(1, N): assembled[i * h - r:i * h + r, :] = v
      for j in range(1, N): assembled[:, j * w - r:j * w + r] = v

    # Calculate MSE and plot
    mse = (delta ** 2).mean()
    imshow(ground_truth, assembled, delta, titles=[
      'Ground Truth', 'Assembled {0}x{0}'.format(N),
      'Difference, MSE = {0:.3f}'.format(mse)], max_cols=max_cols)

  def analyze_rotation(self, key='flattened_phase'):
    # Plot non-rotated image on top left
    images, titles = [getattr(self, key)], ['0 Degree']
    # Rotate, retrieve, and plot
    for k in range(1, 4):
      ig = self.rotate(k)
      images.append(getattr(ig, key))
      titles.append('{} Degree'.format(90 * k))

    # Show images
    imshow(*images, titles=titles, max_cols=2)

  # endregion: Analysis


if __name__ == '__main__':
  import os
  sample = '3t3'
  # sample = 'rbc'
  id = 1
  raw_path = r'../../01-PR/data/{}/sample/{}.tif'.format(sample, id)
  bg_path = r'../../01-PR/data/{}/bg/{}.tif'.format(sample, id)
  if os.path.exists(raw_path):
    assert os.path.exists(bg_path)
    di = Interferogram.imread(raw_path, bg_path, radius=70)
    di.imshow(unwrapped=True)
    di.phase_histogram()
    # di.analyze_windows(4, ignore_gap=0)
  else: print("!! Can not find file '{}'".format(raw_path))



