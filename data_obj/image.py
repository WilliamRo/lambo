import cv2
import numpy as np
import os


class DigitalImage(object):

  def __init__(self, img):
    self.img = img
    self._F = None
    self._Sc = None

  # region: Properties

  @property
  def F(self):
    if self._F is not None: return self._F
    self._F = np.fft.fft2(self.img)
    return self._F

  @property
  def Sc(self):
    """Centralized spectrum"""
    if self._Sc is not None: return self._Sc
    self._Sc = np.abs(np.fft.fftshift(self.F))
    return self._Sc

  # endregion: Properties

  # region: Private Methods

  # endregion: Private Methods

  # region: Public Methods

  def imshow(self, show_fc=False, **kwargs):
    """Show this image
    :param show_fc: whether to show Fourier coefficients
    """
    from lambo.gui.pyplt import imshow
    imgs = [self.img]
    if show_fc: imgs.append(np.log(self.Sc + 1))
    imshow(*imgs, **kwargs)

  # endregion: Public Methods

  # region: Static Methods

  @staticmethod
  def imread(path):
    # TODO: different image types should be tested
    if not os.path.exists(path):
      raise FileExistsError("!! File {} does not exist.")
    img = cv2.imread(path, 0)
    return DigitalImage(img)

  # endregion: Static Methods


if __name__ == '__main__':
  import os
  path = r'../../01-COME/data/3t3/sample/2.tif'
  if os.path.exists(path):
    di = DigitalImage.imread(path)
    di.imshow(True)
  else: print("!! Can not find file '{}'".format(path))

