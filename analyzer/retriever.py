from typing import Callable, List, Optional

from lambo import descartes

from lambo.data_obj.interferogram import Interferogram
from lambo.gui.vinci.vinci import DaVinci
from lambo.misc.local import walk
from lambo.misc.io_utils import save, load

from roma import console
from scipy import signal

import os
import numpy as np


class Retriever(DaVinci):

  def __init__(self, interferograms: List[Interferogram] = (), size: int = 7,
               **kwargs):
    super(Retriever, self).__init__('Retriever', height=size, width=size)

    self.interferograms: List[Interferogram] = list(interferograms)
    self.objects = self.interferograms

    # Protected attributes
    self._save_path = None
    self._radius = None

    # Finalize initialization
    self.keep_3D_view_angle = True
    self._k_space_log = True

  # region: Plot Methods

  def _add_plot(self, extractor: Callable, title=None, plot_3d=False):
    def _plot_2d(x: Interferogram):
      self.imshow_pro(extractor(x), title=title)
    def _plot_3d(x: Interferogram, ax3d):
      self.plot3d(extractor(x), ax3d, title=title)
    self.add_plotter(_plot_3d if plot_3d else _plot_2d)

  def plot_interferogram(self, plot_3d=False):
    self._add_plot(lambda x: x.img, 'Interferogram', plot_3d)
  pi = plot_interferogram

  def plot_bg_unwrapped(self, plot_3d=False):
    def f(x: Interferogram):
      return x.default_background.extracted_angle_unwrapped
    self._add_plot(f, 'Background Phase (Unwrapped)', plot_3d)

  def plot_extracted_image(self, part=None, plot_3d=False):
    f = {None: np.abs, 'r': np.real, 'i': np.imag}
    self._add_plot(
      lambda x: f[part](x.extracted_image),
      'Extracted Image ({})'.format(
        {None: 'magnitude', 'r': 'real', 'i': 'image'}[part]), plot_3d)

  def plot_extracted_phase(self, plot_3d=False):
    self._add_plot(lambda x: x.extracted_angle, 'Extracted Phase', plot_3d)

  def plot_ground_truth(self, plot_3d=False):
    self._add_plot(lambda x: x.flattened_phase, 'Ground Truth', plot_3d)
  pg = plot_ground_truth

  # endregion: Plot Methods

  # region: Commands

  def show_file_path(self):
    ig = self.objects[self.object_cursor]
    assert isinstance(ig, Interferogram)
    console.show_info(
      'File name of interferogram[{}]: `{}`'.format(
        self.object_cursor + 1, ig.get_from_pocket(
          ig.Keys.file_path, default='Not registered')))

    # Print bg path if exists
    bg = ig.default_background
    if bg is not None:
      assert isinstance(bg, Interferogram)
      console.show_info(
        'File name of background[{}]: `{}`'.format(
          self.object_cursor + 1, bg.get_from_pocket(
            ig.Keys.file_path, default='Not registered')))
  fp = show_file_path

  # endregion: Commands

  # region: Showcase

  def show_dual_conv(self, Ls=(19,), omega=45, rs=(0.5, 1.0),
                     show_real_imag=False, rotundity=True, crop=None,
                     show_sum=False, show_extracted=False, combinations=(),
                     max_angle=360, include_origin=False):
    """Syntax of combination: (((a_1, ..., a_p), (r_1, ..., r_q)), ..., ())
    a can be -1, indicating all angles of a
    """

    key = lambda l, a, r: 'DUAL_CONV_{}_L{}_A{}_R{}'.format(
      self.object_cursor, l, a, r)

    def sum(x):
      r, i = [f(x) for f in (np.real, np.imag)]
      return np.abs(r) + np.abs(i)

    def gather(im, knl, l, a, r):
      y = signal.convolve2d(im, knl[::-1, ::-1], mode='same')
      self.put_into_pocket(key(l, a, r), y)

    def extract(x: Interferogram, l, a, r, fmt='abs'):
      if not self.in_pocket(key(l, a, r)):
        # Pre-calculate local match results for current selected obj
        # and L = l
        base_dict = x.get_fourier_dual_basis(
          l, omega, rs, True, rotundity=rotundity,
          include_origin=include_origin, max_angle=max_angle)

        # Add combination to list if required
        for comb in combinations:
          assert isinstance(comb, (tuple, list)) and len(comb) == 2
          _as, _rs = comb
          if _as == -1: _as = [a for a in range(0, max_angle, omega)]
          base_dict[comb] = np.sum(
            [base_dict[(a, r)] for a in _as for r in _rs], axis=0)

        total = len(base_dict)

        # Get image to be convolved
        im = self.crop(x.img, crop)

        for i, ((_a, _r), b) in enumerate(base_dict.items()):
          self.show_status(
            f'[{i+1}/{total}] Convolving obj {self.object_cursor + 1} with ' 
            f'{"round" if rotundity else "square"} kernel of size {l}, '
            f'(a={_a}, r={_r:.2f}) ...')
          gather(im, b, l, _a, _r)

      return {'abs': np.abs, 'r': np.real, 'i': np.imag, 'sum': sum}[fmt](
        self.get_from_pocket(key(l, a, r), key_should_exist=True))

    #
    self._add_plot(lambda x: self.crop(x.img, crop), 'Raw interferogram')
    if show_extracted:
      self._add_plot(lambda x: self.crop(np.abs(x.extracted_image), crop),
                     'Extracted Intensity')
      self._add_plot(lambda x: self.crop(np.real(x.extracted_image), crop),
                     'Extracted (real)')
      self._add_plot(lambda x: self.crop(np.imag(x.extracted_image), crop),
                     'Extracted (imag)')
      self._add_plot(lambda x: self.crop(x.extracted_angle, crop),
                     'Extracted Phase')

    #
    for l in Ls:
      _add = lambda _r, _a, _l=l, fmt='abs': self._add_plot(
        extractor=lambda x: extract(x, _l, _a, _r, fmt),
        title='L = {}, a = {}, r = {:.2f} ({}, {})'.format(
          _l, _a, _r, {'abs': 'abs', 'r': 'real part', 'i': 'image part',
                       'sum': 'sum'}[fmt],
          'round' if rotundity else 'square'))

      def _add_group(r, a):
        _add(r, a, fmt='abs')
        if show_sum: _add(r, a, fmt='sum')
        if show_real_imag:
          _add(r, a, fmt='r')
          _add(r, a, fmt='i')

      if include_origin: _add_group(0, 0)
      for r, a in descartes(rs, range(0, max_angle, omega)): _add_group(r, a)
      for comb in combinations: _add_group(comb[1], comb[0])

    # Add
    self._add_plot(lambda x: self.crop(x.flattened_phase, crop), 'Ground Truth')

    #
    self.show()

  # endregion: Showcase

  # region: Public Methods

  @staticmethod
  def crop(img: np.ndarray, R=None):
    if R is None: return img
    assert isinstance(R, int) and R > 10
    ci, cj = [s // 2 for s in img.shape]
    return img[ci-R:ci+R, cj-R:cj+R]

  @staticmethod
  def save(interferograms: List[Interferogram], save_path,
           keys=('flattened_phase',), overwrite=False):
    total = len(interferograms)
    console.show_status('Generating {} ...'.format(', '.join(
      ['`{}`'.format(k) for k in keys])))
    for i, ig in enumerate(interferograms):
      console.print_progress(i, total)
      for key in keys: _ = getattr(ig, key)
      # Localize intermediate results so that they can be saved by pickle
      ig.localize('extracted_image')
      ig.localize('corrected_image')
      ig.localize('flattened_phase')
    if os.path.exists(save_path) and not overwrite:
      raise FileExistsError('!! File `{}` already exist.'.format(save_path))
    save(interferograms, save_path)
    console.show_status('Interferograms saved to `{}`'.format(save_path))

  @classmethod
  def initialize(cls, path, radius=None, seq_id=1, save=True):
    save_keys = ['extracted_image', 'corrected_image', 'flattened_phase']
    suffix = '({})'.format('-'.join([k[:2] for k in save_keys]))
    r = cls(Retriever.read_interferograms(
      path, radius, seq_id, save=save, suffix=suffix, save_keys=save_keys))
    return r

  @classmethod
  def get_save_file_name(cls, radius, seq_id, suffix=''):
    return 'ig-rad{}-seq{}{}.save'.format(radius, seq_id, suffix)

  @classmethod
  def read_interferograms(
      cls, path: str, radius=None, seq_id: int = 1, save=False,
      save_keys=('flattened_phase',), suffix='') -> List[Interferogram]:
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
    :param save: whether to save interferogram list
    :param save_keys: contents to be localized
    :param suffix: file suffix
    :return: a list of interferograms
    """
    # Read directly if buffer file exist
    save_fn = cls.get_save_file_name(radius, seq_id, suffix)
    save_path = os.path.join(path, save_fn)
    if radius is not None and os.path.exists(save_path):
      console.show_status('Loading `{}` ...'.format(save_path))
      return load(save_path)

    # Get sample name
    sample_name = os.path.basename(path)

    # Find the file organization type by looking at the sub-folders
    interferograms = []
    subfolders = walk(path, type_filter='folder', return_basename=True)
    # Remove folders which should be ignored
    if 'trash' in subfolders: subfolders.remove('trash')

    if 'sample' in subfolders and 'bg' in subfolders:
      # Case (1)
      console.show_status('Reading files organized as `sample/bg` ...')
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
      console.show_status('Reading files organized as `odd/even` ...')
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
        'Reading files from `{}` organized as video sequences ...'.format(path))

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
    for ig in interferograms: ig.sample_token = sample_name
    console.show_status('{} interferograms have been read.'.format(
      len(interferograms)))

    # Save if required
    if radius is not None and save:
      Retriever.save(interferograms, save_path, save_keys, overwrite=False)

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
  r.plot_extracted_phase(True)
  # r.plot
  # r.plot_bg_unwrapped()
  r.plot_bg_unwrapped(True)
  r.plot_ground_truth()
  r.show()

