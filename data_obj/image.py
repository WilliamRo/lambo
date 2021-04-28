import cv2
import numpy as np
import os

from lambo.abstract.noear import Nomear
from lambo.gui.pyplt import imshow

from skimage.transform import rotate


class DigitalImage(Nomear):

  def __init__(self, img: np.ndarray, **kwargs):
    self.img = img
    self._init_kwargs = kwargs

  # region: Properties

  @property
  def F(self) -> np.ndarray:
    return self.get_from_pocket(
      'fft2(x)', initializer=lambda: np.fft.fft2(self.img))

  @property
  def Fc(self) -> np.ndarray:
    return self.get_from_pocket(
      'fftshift(fft2(x))', initializer=lambda: np.fft.fftshift(self.F))

  @property
  def Sc(self) -> np.ndarray:
    """Centralized spectrum"""
    return self.get_from_pocket(
      'abs(fftshift(fft2(x)))', initializer=lambda: np.abs(self.Fc))

  @property
  def size(self):
    return self.img.shape

  # endregion: Properties

  # region: Private Methods

  # endregion: Private Methods

  # region: Public Methods

  def imshow(self, show_fc=False, **kwargs):
    """Show this image
    :param show_fc: whether to show Fourier coefficients
    """
    imgs = [self.img]
    if show_fc: imgs.append(np.log(self.Sc + 1))
    imshow(*imgs, **kwargs)

  # endregion: Public Methods

  # region: Class Methods

  @classmethod
  def imread(cls, path, return_array=False, **kwargs):
    # TODO: different image types should be tested
    if not os.path.exists(path):
      raise FileExistsError("!! File {} does not exist.")
    img = cv2.imread(path, 0)
    if return_array: return img
    return cls(img, **kwargs)

  # endregion: Class Methods

  # region: Image Transformation

  @staticmethod
  def rotate_image(x: np.ndarray, angle: float, resize: bool = False,
                   mode='constant'):
    return rotate(x, angle, resize, mode=mode)

  @staticmethod
  def get_downtown_area(x: np.ndarray, p2=0):
    """All pixels in `downtown` area of an rotated image are indigenous"""
    h, w = x.shape[:2]
    ci, cj = h // 2, w // 2
    # radius of the in-circle
    r = min(ci, cj)
    d = int(np.floor(r / np.sqrt(2)))
    # Make sure output side-length is integer multiples of (2 ** p2)
    unit = 2 ** p2
    d = d // unit * unit
    return x[ci-d:ci+d, cj-d:cj+d]

  # endregion: Image Transformation


