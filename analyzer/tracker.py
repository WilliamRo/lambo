from lambo.gui.vinci.vinci import DaVinci

from lambo.misc.local import walk
from pandas import DataFrame, Series
from roma import console
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as p
import pims
import time
import trackpy as tp
import warnings


class Tracker(DaVinci):

  class Keys(DaVinci.Keys):
    locations = 'LOCATIONS'
    trajectories = 'TRAJECTORIES'
    n_frames = 'N_FRAMES'
    show_titles = 'SHOW_TITLES'


  def __init__(self, frames, size: int = 5, **kwargs):
    # Call parent's constructor
    super(Tracker, self).__init__('Nuclei Browser', height=size, width=size)

    # Specific attributes
    self.raw_frames = frames
    self.images: np.ndarray = frames / np.max(frames)
    self.objects = self.images
    self.layer_plotter = [self.imshow]

    # Attributes for display
    self.kwargs = kwargs

    # Suppress all warnings
    warnings.filterwarnings('ignore')

    # Configurations
    self.locate_configs = {}
    self.link_configs = {}

    self.effective_locate_config = {}
    self.effective_link_config = {}

  # region: Properties

  @property
  def show_titles(self):
    return self.get_from_pocket(
      self.Keys.show_titles, initializer=lambda: False)

  @property
  def file_name(self):
    return getattr(self.raw_frames, '_filename')

  @property
  def n_frames(self):
    return self.get_from_pocket(self.Keys.n_frames, len(self.images))

  @n_frames.setter
  def n_frames(self, val: int):
    self.objects = self.images[:val]
    self.put_into_pocket(self.Keys.n_frames, val)

  @property
  def locations(self) -> DataFrame:
    return self.get_from_pocket(self.Keys.locations, None)

  @locations.setter
  def locations(self, val: DataFrame):
    self.put_into_pocket(self.Keys.locations, val, exclusive=False)

  @property
  def trajectories(self):
    return self.get_from_pocket(self.Keys.trajectories, None)

  @trajectories.setter
  def trajectories(self, val):
    self.put_into_pocket(self.Keys.trajectories, val, exclusive=False)

  # endregion: Properties

  # region: Static Methods

  @staticmethod
  def read(file_path: str, show_info: bool = False, **kwargs):
    if not os.path.exists(file_path):
      raise FileNotFoundError('!! File `{}` not found.'.format(file_path))
    frames = pims.open(file_path)
    if show_info:
      console.show_info('Data info:')
      console.split()
      print(frames)
      print('Range: [{}, {}]'.format(np.min(frames), np.max(frames)))
      console.split()
    return Tracker(frames, **kwargs)

  @classmethod
  def read_by_index(cls, data_dir: str, index: Optional[int] = None,
                    pattern: str = '*.tif', show_info: bool = False, **kwargs):
    if not os.path.exists(data_dir):
      raise FileNotFoundError('!! Directory `{}` not found.'.format(data_dir))
    ls = not (isinstance(index, int) and index > 0)
    file_list = walk(data_dir, 'file', pattern, return_basename=ls)
    if ls:
      for i, fn in enumerate(file_list): print('[{}] {}'.format(i + 1, fn))
    else:
      return cls.read(file_list[index + 1], show_info=show_info, **kwargs)

  # endregion: Static Methods

  # region: Setting

  def get_update_function(self, config_dict: dict):
    def update(key, value):
      if value is None: return
      config_dict[key] = value
    return update

  def config_locate(self,
                    diameter: int = None,
                    minmass: float = None,
                    maxsize: float = None,
                    separation: float = None,
                    noise_size: float = None,
                    smoothing_size: float = None,
                    threshold: float = None,
                    invert: bool = False,
                    percentile: float = None,
                    topn: int = None,
                    preprocess: bool = None,
                    max_iterations: int = None,
                    characterize: bool = None,
                    engine: str = None):
    # Sanity check
    assert engine in (None, 'auto', 'python', 'numba')

    # Update locate_config accordingly
    update = self.get_update_function(self.locate_configs)
    update('diameter', diameter)
    update('minmass', minmass)
    update('maxsize', maxsize)
    update('separation', separation)
    update('noise_size', noise_size)
    update('smoothing_size', smoothing_size)
    update('threshold', threshold)
    update('invert', invert)
    update('percentile', percentile)
    update('topn', topn)
    update('preprocess', preprocess)
    update('max_iterations', max_iterations)
    update('characterize', characterize)
    update('engine', engine)

  def config_link(self,
                  search_range: float = None,
                  memory: int = None,
                  adaptive_stop: float = None,
                  adaptive_step: float = None,
                  neighbor_strategy: str = None,
                  link_strategy: str = None):

    # Sanity check
    assert neighbor_strategy in ('KDTree', 'BTree', None)
    assert link_strategy in (
      'recursive', 'nonrecursive', 'numba', 'hybrid', 'drop', 'auto', None)

    # Update link_config accordingly
    update = self.get_update_function(self.link_configs)
    update('search_range', search_range)
    update('memory', memory)
    update('adaptive_stop', adaptive_stop)
    update('adaptive_step', adaptive_step)
    update('neighbor_strategy', neighbor_strategy)
    update('link_strategy', link_strategy)

  # endregion: Setting

  # region: Visualization

  def show_locations(self, show_traj=False, **locate_configs):
    # TODO: for some reason, locate config can be modified here
    # self.locate_configs has the highest priority
    locate_configs.update(self.locate_configs)

    # Calculate location if not exist
    if self.locations is None:
      self.locate(plot_progress=True, **locate_configs)
    configs = self.effective_locate_config
    df = self.locations

    # Link location if necessary
    if show_traj:
      if self.trajectories is None: self.link()
      configs = self.effective_link_config
      df = self.trajectories

    # Display
    df = df[df['frame'] == self.object_cursor]
    tp.annotate(df, self.raw_frames[self.object_cursor], ax=self.axes)

    # Set title
    title = None
    if self.show_titles:
      title = ', '.join(['{} = {}'.format(k, v) for k, v in configs.items()])
      title += ' (#{})'.format(df.size)
    self.set_im_axes(title=title)

    return self.locations

  # endregion: Visualization

  # region: Analysis

  def locate(self, diameter=7, plot_progress=False, **configs):
    configs['diameter'] = diameter

    # self.locate_configs has the highest priority
    configs.update(self.locate_configs)

    # Calculate locations
    tic = time.time()

    # Locate
    tp.quiet()

    def after_locate(frame_no, features):
      # Plot progress if required
      if plot_progress:
        self.show_text(self.axes, 'Calculating {}/{} ...'.format(
          frame_no, self.n_frames))
        self.canvas.draw()
      console.print_progress(frame_no, self.n_frames)
      return features

    self.locations = tp.batch(
      self.objects, processes=0, after_locate=after_locate, **configs)

    console.show_status('Locating completed. Configurations:')
    console.supplement(configs)

    # Clear status
    if plot_progress: self.axes.cla()

    self.effective_locate_config = configs

  def link(self, search_range: int = 5, memory: int = 2, **configs):
    # Make sure particles have been located
    if self.locations is None:
      print(' ! No locations found.')
      return

    configs['search_range'] = search_range
    configs['memory'] = memory

    # self.link_config has the highest priority
    configs.update(self.link_configs)

    # Link locations
    self.trajectories = tp.link(self.locations, **configs)
    console.show_status('Linking completed. Configurations:')
    console.supplement(configs)

    self.effective_link_config = configs

    self._draw()

  def filter_traj(self, threshold: int):
    self.trajectories = tp.filter_stubs(self.trajectories, threshold)
    self.effective_link_config['threshold'] = threshold

    self._draw()

  # endregion: Analysis

  # region: Builtin Commands

  def set_title(self, flag: int = 1):
    assert flag in (0, 1)
    self.put_into_pocket(self.Keys.show_titles, flag, exclusive=False)
    self._draw()
  st = set_title

  # endregion: Builtin Commands


if __name__ == '__main__':
  data_dir = r'E:\lambai\10-NR\data\mar2021'

  index = 2
  diameter = 19
  minmass = 0.9

  tk = Tracker.read_by_index(data_dir, index, show_info=True)
  # tk.n_frames = 10
  tk.config_locate(diameter=diameter, minmass=minmass)
  # tk.config_link(search_range=10, memory=0)

  tk.add_plotter(tk.imshow)
  # tk.add_plotter(tk.histogram)
  tk.add_plotter(tk.show_locations)
  # tk.add_plotter(lambda: tk.show_locations(show_traj=True))
  tk.show()
