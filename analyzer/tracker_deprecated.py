from lambo.abstract.noear import Nomear
from lambo.misc.local import walk
from pandas import DataFrame, Series
from roma import console
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pims
import time
import trackpy as tp
import warnings


class Tracker(Nomear):

  class Keys:
    figure = 'FIGURE'
    axes = 'AXES'

    locations = 'LOCATIONS'
    n_frames = 'N_FRAMES'

  def __init__(self, frames, size: int = 5, **kwargs):
    self.raw_frames = frames
    self.images = frames / np.max(frames)

    # Attributes for display
    self._size = size
    self._refresh = None
    self._cursor = 0
    self._layer_cursor = 0
    self.layers = [self.show_raw_frames]

    self.kwargs = kwargs

    # Suppress all warnings
    warnings.filterwarnings('ignore')

  # region: Properties

  @property
  def file_name(self):
    return getattr(self.raw_frames, '_filename')

  @property
  def cursor(self) -> int:
    return self._cursor

  @cursor.setter
  def cursor(self, val: int):
    self._cursor = val % self.n_frames

  @property
  def cursor_str(self):
    return '[{}/{}]'.format(self.cursor + 1, self.n_frames)

  @property
  def n_frames(self):
    return self.get_from_pocket(self.Keys.n_frames, len(self.images))

  @n_frames.setter
  def n_frames(self, val):
    self.put_into_pocket(self.Keys.n_frames, val)

  @property
  def figure(self) -> plt.Figure:
    if self.Keys.figure not in self._pocket:
      self.figure = plt.figure(figsize=(self._size, self._size))
    return self.get_from_pocket(self.Keys.figure)

  @property
  def canvas(self) -> plt.FigureCanvasBase:
    return self.figure.canvas

  @figure.setter
  def figure(self, val: plt.Figure):
    self.put_into_pocket(self.Keys.figure, val)

  @property
  def axes(self) -> plt.Axes:
    if self.Keys.axes not in self._pocket:
      self.axes = self.figure.add_subplot(111)
    return self.get_from_pocket(self.Keys.axes)

  @axes.setter
  def axes(self, val: plt.Axes):
    self.put_into_pocket(self.Keys.axes, val)

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

  # region: Visualization

  def _default_key_event(self, event):
    key = event.key
    if key in ('escape', 'q'):
      plt.close('all')
      return
    elif key in ('j', 'right'):
      self.cursor += 1
    elif key in ('k', 'left'):
      self.cursor -= 1
    elif key in ('h', 'down'):
      self._layer_cursor -= 1
    elif key in ('l', 'up'):
      self._layer_cursor += 1
    else:
      print('>> key "{}" pressed'.format(event.key))
      return
    self.refresh()

  def initialize_plot(self, key_press_event: Optional[Callable] = None):
    self.canvas.set_window_title(self.file_name)
    self.canvas.mpl_disconnect(self.canvas.manager.key_press_handler_id)

    # Bind events if provided
    if key_press_event is not None:
      assert callable(key_press_event)
      plt.connect('key_press_event', key_press_event)

  def refresh(self):
    if not self.layers: return
    # Clear axes
    self.axes.cla()
    # Get layer cursor
    layer_cursor = self._layer_cursor % len(self.layers)

    # Call layer method
    self.layers[layer_cursor]()

    # Tight layout and draw
    self.figure.tight_layout()
    self.canvas.draw()

  def view(self, *layer_methods):
    self.initialize_plot(self._default_key_event)
    # Set refresh function if provided
    for method in layer_methods: self.layers.append(method)
    # Refresh and show
    self.refresh()
    plt.show()

  # endregion: Visualization

  # region: Layers

  # region: Toolbox

  def set_axes_style(self, title=''):
    self.axes.set_axis_off()
    self.axes.set_title('[{}/{}] {}'.format(
      self.cursor + 1, self.n_frames, title))

  def show_text(self, text):
    self.axes.cla()
    self.axes.text(0.5, 0.5, text, ha='center', va='center')
    self.axes.set_axis_off()
    self.canvas.draw()

  # endregion: Toolbox

  def show_raw_frames(self):
    self.axes.imshow(self.images[self.cursor])
    self.set_axes_style()

  def raw_frames_hist(self):
    pixels = np.ravel(self.images[self.cursor])
    self.axes.hist(pixels, bins=20, density=True)
    self.axes.set_xlim(0, 1)
    self.axes.set_ylim(0, 20)
    self.axes.set_aspect('auto')
    self.axes.set_title(self.cursor_str + ' Histogram of raw image')

  def locate(self, diameter, **kwargs):
    kwargs.get('minmass', None)
    kwargs.get('maxsize', None)
    kwargs.get('separation', None)
    kwargs.get('noise_size', None)
    kwargs.get('smooth_size', None)
    kwargs.get('threshold', None)
    kwargs.get('invert', None)
    kwargs.get('percentile', None)
    kwargs.get('topn', None)

    def method():
      if self.Keys.locations not in self._pocket:
        # Calculate locations
        data_frames = []
        tic = time.time()
        for i in range(self.n_frames):
          self.show_text('Calculating {}/{} ...'.format(i + 1, self.n_frames))
          df = tp.locate(self.images[i], diameter, **kwargs)
          data_frames.append(df)
          console.print_progress(i + 1, self.n_frames, start_time=tic)

        self.put_into_pocket(self.Keys.locations, data_frames, exclusive=False)
        console.show_status('Locating completed. Configurations:')
        kwargs['diameter'] = diameter
        console.supplement(kwargs)

        # Clear status
        self.axes.cla()

      # Display
      dfs = self.get_from_pocket(self.Keys.locations)
      tp.annotate(dfs[self.cursor], self.raw_frames[self.cursor], ax=self.axes)
      # Set title
      title = ', '.join(['{} = {}'.format(k, v) for k, v in kwargs.items()])
      self.set_axes_style(title)

    return method

  # endregion: Layers

  # region: Refresh Methods

  def locate_probe(self, diameter, n_frames=None, **kwargs):
    pass

  # endregion: Refresh Methods


if __name__ == '__main__':
  data_dir = r'E:\lambai\10-NR\data\mar2021'

  index = 2
  diameter = 7
  minmass = 0.2

  tk = Tracker.read_by_index(data_dir, index, show_info=True)
  # tk.n_frames = 5
  tk.view(
    tk.raw_frames_hist,
    tk.locate(diameter=diameter, minmass=minmass),
  )
