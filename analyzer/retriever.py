from typing import Callable, List, Optional

from lambo.data_obj.interferogram import Interferogram
from lambo.gui.vinci.vinci import DaVinci
from lambo.misc.local import walk
from lambo.misc.io_utils import save, load

from roma import console

import os


class Retriever(DaVinci):

  def __init__(self, interferograms: List[Interferogram] = (), size: int = 7,
               **kwargs):
    super(Retriever, self).__init__('Retriever', height=size, width=size)

    self.interferograms: List[Interferogram] = list(interferograms)
    self.objects = self.interferograms

    # Protected attributes
    self._save_path = None
    self._radius = None
    self._color_bar = False

    # Finalize initialization
    self.keep_3D_view_angle = True
    self.state_machine.register_key_event('c', self.toggle_color_bar)

  # region: Plot Methods

  def _add_plot(self, extractor: Callable, title=None, plot_3d=False):
    def _plot_2d(x: Interferogram, ax):
      self.imshow(extractor(x), ax, title=title, color_bar=self._color_bar)
    def _plot_3d(x: Interferogram, ax3d):
      self.plot3d(extractor(x), ax3d, title=title)
    self.add_plotter(_plot_3d if plot_3d else _plot_2d)

  def plot_interferogram(self, plot_3d=False):
    self._add_plot(lambda x: x.img, 'Interferogram', plot_3d)
  pi = plot_interferogram

  def plot_extracted_phase(self, plot_3d=False):
    self._add_plot(lambda x: x.extracted_angle, 'Extracted Phase', plot_3d)

  def plot_ground_truth(self, plot_3d=False):
    self._add_plot(lambda x: x.flattened_phase, 'Ground Truth', plot_3d)
  pg = plot_ground_truth

  # endregion: Plot Methods

  # region: Commands

  def toggle_color_bar(self):
    self._color_bar = not  self._color_bar
    self._draw()
  tcb = toggle_color_bar

  # endregion: Commands

  # region: Public Methods

  def save(self, overwrite=False, keys=()):
    assert self._save_path is not None

    total = len(self.interferograms)
    console.show_status('Traversing keys ...')
    for i, ig in enumerate(self.interferograms):
      console.print_progress(i, total)
      # Localize pocket so that it can be saved by pickle
      ig.localize()
      _ = ig.flattened_phase
      for key in keys: _ = getattr(ig, key)
    if os.path.exists(self._save_path) and not overwrite:
      raise FileExistsError('!! File `{}` already exist.'.format(
        self._save_path))
    save(self.interferograms, self._save_path)
    console.show_status('Interferograms saved to `{}`'.format(self._save_path))

  @classmethod
  def initialize(cls, path, radius=None, seq_id=1, save=True):
    r = cls(Retriever.read_interferograms(path, radius, seq_id))
    if radius is not None:
      r._save_path = os.path.join(path, cls.get_save_file_name(radius, seq_id))
      if not os.path.exists(r._save_path) and save:
        r.save(overwrite=False, keys=['extracted_angle_unwrapped'])
    return r

  @classmethod
  def get_save_file_name(cls, radius, seq_id):
    assert radius is not None
    return 'ig-{}-{}.save'.format(radius, seq_id)

  @classmethod
  def read_interferograms(
      cls, path: str, radius=None, seq_id: int = 1) -> List[Interferogram]:
    """Read a list of interferograms inside the given path. Raw images along
    with their corresponding background images for calibration should be
    organized in one of the three ways listed below:
    (1) path
          |-sample
             |-1.tif
             |- ...
             |-k.tif     # k-th interferogram with sample
             |- ...
          |-bg
             |-1.tif     # background of k-th interferogram
             |- ...
             |-k.tif
             |- ...

    (2) path
          |-1.tif
          |-2.tif
          |- ...
          |-<2k-1>.tif   # k-th interferogram with sample
          |-<2k>.tif     # background of k-th interferogram
          |- ...

    (3) path
          |-1
          |-2
          |- ...
          |-p            # p-th sequence
            |-1.tif
            |- ...
            |-k.tif      # k-th interferogram with sample of p-th sequence
            |- ...
            |-bg.tif     # common background interferogram of p-th sequence
          |-...

    :param path: Trial path
    :param radius: filter radius in k-space
    :param seq_id: if interferograms are organized as sequences, only the
                   `seq_id`-th sequence will be read
    :return: a list of interferograms
    """
    # Read directly if buffer file exist
    if radius is not None:
      save_fn = cls.get_save_file_name(radius, seq_id)
      save_path = os.path.join(path, save_fn)
      if os.path.exists(save_path):
        console.show_status('Loading `{}` ...'.format(save_path))
        return load(save_path)

    # Find the file organization type by looking at the sub-folders
    interferograms = []
    subfolders = walk(path, type_filter='folder', return_basename=True)
    # Remove folders which should be ignored
    if 'trash' in subfolders: subfolders.remove('trash')
    if 'sample' in subfolders and 'bg' in subfolders:
      # Case (1)
      console.show_status('Reading files (organization type 1) ...')
      sample_folder, bg_folder = [
        os.path.join(path, fn) for fn in ('sample', 'bg')]
      # Scan all sample files in sample folder
      for sample_path in walk(
          sample_folder, type_filter='file', pattern='*.tif*'):
        fn = os.path.basename(sample_path)
        bg_path = os.path.join(bg_folder, fn)
        if not os.path.exists(bg_path): console.warning(
          ' ! Background file `{}` does not exist'.format(bg_path))
        else:
          ig = Interferogram.imread(sample_path, bg_path, radius)
          interferograms.append(ig)
    elif len(subfolders) == 0:
      # Case (2)
      console.show_status('Reading files (organization type 2) ...')
      file_list = walk(
        path, type_filter='file', pattern='*.tif', return_basename=True)
      while len(file_list) > 0:
        fn = file_list.pop(0)
        # Get int id
        index = int(fn.split('.')[0])
        # If file is sample
        if index % 2 == 1:
          sample_fn = fn
          bg_fn = '{}.tif'.format(index + 1)
          mate = bg_fn
        else:
          bg_fn = fn
          sample_fn = '{}.tif'.format(index - 1)
          mate = sample_fn
        # Check if mate exists
        if mate not in file_list:
          console.warning(' ! Mate file `{}` of `{}` does not exist'.format(
            mate, fn))
          continue
        # Remove mate from list and append sample/bg path to pairs
        file_list.remove(mate)
        # Read interferogram
        ig = Interferogram.imread(
          *[os.path.join(path, f) for f in (sample_fn, bg_fn)], radius)
        interferograms.append(ig)
    else:
      # Case (3)
      assert isinstance(seq_id, int) and seq_id > 0
      if str(seq_id) not in subfolders:
        console.warning('`seq_id` should be one of the following options:')
        for sf in subfolders: console.supplement(sf, color='red')
        raise FileNotFoundError
      path = os.path.join(path, str(seq_id))
      console.show_status(
        'Reading files from `{}` (organization type 3) ...'.format(path))

      # Read background
      bg = Interferogram.imread(os.path.join(path, 'bg.tif'), radius=radius)
      total = len(walk(path, 'file', pattern='*.tif')) - 1
      for i in range(total):
        console.print_progress(i, total)
        p = os.path.join(path, '{}.tif'.format(i + 1))
        ig = Interferogram.imread(p, radius=radius)
        # Using peak_index from background
        ig.peak_index = bg.peak_index
        # Set the same background to all interferograms
        ig._backgrounds = [bg]
        interferograms.append(ig)

    # Finalize
    console.show_status('{} interferograms have been read.'.format(
      len(interferograms)))
    return interferograms

  # endregion: Public Methods


if __name__ == '__main__':
  trial_root = r'E:\lambai\01-PR\data'
  trial_name = '01-3t3'
  # trial_name = '80-spacer-0526'
  path = os.path.join(trial_root, trial_name)

  r = Retriever.initialize(path, 80)
  # r.save(overwrite=True)
  r.plot_interferogram()
  r.plot_ground_truth(True)
  r.show()

