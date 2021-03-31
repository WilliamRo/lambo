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
  def flattened_phase(self) -> np.ndarray:
    def _flattened_phase():
      phase = self.unwrapped_phase.copy()
      if callable(fit_plane):
        flattened = phase - fit_plane(phase, radius=0.01)
        flattened = np.maximum(flattened, 0)
        return flattened
      # TODO: consider using a better algorithm
      P, Q = 30, 100
      H, W = phase.shape
      x_conf = np.mean(phase[:, -P] - phase[:, P])
      y_conf = np.mean(phase[-P, :] - phase[P, :])
      x_conf_vec = np.linspace(x_conf, 0, W)
      y_conf_vec = np.linspace(y_conf, 0, H).reshape(H, 1)
      phase += x_conf_vec + y_conf_vec
      phase -= np.mean(phase[0:Q, 0:Q])  # mean or median
      phase = np.maximum(phase, 0)
      return phase
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

  def imshow(self, show_img=True, show_fc=False, circle=False, extracted=False,
             show_angle=False, show_unwrapped=False, show_flattened=False,
             show_sample_mask=False, **kwargs):
    """Show this image
    :param show_fc: whether to show Fourier coefficients
    """
    imgs = []
    # (0) interferogram
    if show_img: imgs.append(self.img)
    # (1) spectrum
    if show_fc:
      im = np.log(self.Sc + 1)
      if circle: im = (im, lambda: plt.gca().add_artist(plt.Circle(
        list(reversed(self.peak_index)), self.radius, color='r', fill=False)))
      imgs.append(im)
    # (2) extracted image
    if extracted:
      imgs.append(np.abs(self.extracted_image))
    # (3) show retrieved phase
    if show_angle: imgs.append(self.delta_angle)
    # (4) Unwrapped phase image
    if show_unwrapped: imgs.append(self.unwrapped_phase)
    # (5) Balanced phase image
    if show_flattened: imgs.append(self.flattened_phase)
    # (6) Sample mask
    if show_sample_mask: imgs.append(self.soft_mask())

    # Show image using lambo.imshow
    imshow(*imgs, **kwargs)

  def phase_histogram(self):
    array = np.ravel(self.flattened_phase)
    print('Range: [{:.2f}, {:.2f}]'.format(np.min(array), np.max(array)))
    plt.hist(array)
    bind_quick_close()
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

  # endregion: Analysis


if __name__ == '__main__':
  import os
  sample = '3t3'
  # sample = 'rbc'
  id = 2
  raw_path = r'../../01-PR/data/{}/sample/{}.tif'.format(sample, id)
  bg_path = r'../../01-PR/data/{}/bg/{}.tif'.format(sample, id)
  if os.path.exists(raw_path):
    assert os.path.exists(bg_path)
    di = Interferogram.imread(raw_path, bg_path, radius=70)
    # di.imshow(show_unwrapped=True)
    # di.phase_histogram()
    di.analyze_windows(4, ignore_gap=0)
  else: print("!! Can not find file '{}'".format(raw_path))



