import cv2
import numpy as np
import os

from lambo.abstract.noear import Nomear
from lambo.misc.local import walk

from roma import console


class Video(Nomear):

  def __init__(self, img_seq: np.ndarray, **kwargs):
    self._img_seq = img_seq
    self._init_kwargs = kwargs

  # region: Properties

  @property
  def sequence(self) -> np.ndarray:
    return self._img_seq

  # endregion: Properties

  # region: Private Methods

  # endregion: Private Methods

  # region: Public Methods

  @staticmethod
  def read(path, fmt=None):
    if not os.path.exists(path):
      raise FileExistsError('!! File `{}` does not exist.'.format(path))
    # Read image all frames in specified .tif file
    _, images = cv2.imreadmulti(path)
    img_seq = np.stack(images)
    # Process data according to given fmt
    if fmt in ('grey', 'duplicated_channel'): img_seq = img_seq[..., 0]
    console.show_status('Read `{}`, data shape: {}'.format(
      os.path.basename(path), img_seq.shape))
    # Wrap data into Video object and return
    return Video(img_seq)

  def tfr_view(self):
    """View image sequence using tframe.ImageViewer"""
    from tframe.data.images.image_viewer import ImageViewer
    from tframe.data.dataset import DataSet

    images = self.sequence
    images = images / np.max(images)
    ds = DataSet(features=images)
    viewer = ImageViewer(ds)
    viewer.show()

  # endregion: Public Methods


if __name__ == '__main__':
  data_dir = r'../../10-NR/data/mar2021/'
  file_list = walk(data_dir, type_filter='file', pattern='*.tif')
  index = 6
  v = Video.read(file_list[index])
  v.tfr_view()

