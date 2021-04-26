from matplotlib.gridspec import GridSpec
from matplotlib import cm

from lambo.abstract.noear import Nomear

import inspect
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Board(Nomear):
  """
  Developer's Notes
  21-Apr-23
  Board knows nothing about StateMachine
  """

  class Configs:
    show_after_ms = 1

  class Keys:
    figure = 'FIGURE'
    axes = 'AXES'

  def __init__(self, title=None, height=5, width=5):
    self.objects = []
    self.layer_plotter = []

    self._object_cursor = 0
    self._layer_cursor = 0

    # Other attributes
    self.title = title
    self.fig_height = height
    self.fig_width = width

  # region: Properties

  @property
  def object_cursor(self):
    return self._object_cursor

  @object_cursor.setter
  def object_cursor(self, val):
    if not self.objects: return
    previous_cursor = self.object_cursor
    self._object_cursor = val % len(self.objects)
    if self.object_cursor != previous_cursor: self._draw()

  @property
  def layer_cursor(self):
    return self._layer_cursor

  @layer_cursor.setter
  def layer_cursor(self, val):
    if not self.layer_plotter: return
    previous_cursor = self.layer_cursor
    self._layer_cursor = val % len(self.layer_plotter)
    if self.layer_cursor != previous_cursor: self._draw()

  @property
  def current_plotter(self):
    if not self.layer_plotter: return self.show_text
    return self.layer_plotter[self.layer_cursor]

  @property
  def figure(self) -> plt.Figure:
    init_fig = lambda: plt.figure(figsize=(self.fig_height, self.fig_width))
    return self.get_from_pocket(self.Keys.figure, initializer=init_fig)

  @property
  def canvas(self) -> plt.FigureCanvasBase:
    return self.figure.canvas

  @property
  def window(self):
    return self.canvas.manager.window

  @property
  def axes(self) -> plt.Axes:
    init_axes = lambda: self.figure.add_subplot(111)
    return self.get_from_pocket(self.Keys.axes, initializer=init_axes)

  @property
  def backend(self) -> str:
    return matplotlib.get_backend()

  @property
  def backend_is_TkAgg(self):
    return self.backend == 'TkAgg'

  @property
  def backend_is_WXAgg(self):
    return self.backend == 'WXAgg'

  @property
  def win_title(self):
    result = ''
    if len(self.objects) > 0:
      result = '[{}/{}]'.format(self.object_cursor + 1, len(self.objects))
    if self.title is not None:
      result += ' ' + self.title
    if not result: return 'Untitled'
    return result

  # endregion: Properties

  # region: Private Methods

  def _draw(self):
    """Draw stuff.
    Case 1:
    Case 2:
    """
    # Clear all
    self._clear()

    # Set windows title
    self.canvas.set_window_title(self.win_title)
    # Get arguments for current plotter
    kwargs = self._get_kwargs_for_plotter(self.current_plotter)

    # Plot
    self.current_plotter(**kwargs)

    # Tight layout and refresh
    plt.tight_layout()
    self.canvas.draw()

  def _clear(self):
    self.axes.cla()

  def _get_kwargs_for_plotter(self, plotter):
    assert callable(plotter)
    # Get method signature
    sig = inspect.signature(plotter).parameters
    # Get key-word arguments for method according to its signature
    kwargs = {}
    for kw in sig.keys():
      if kw in ('obj', 'x', 'img'):
        kwargs[kw] = self.objects[self.object_cursor]
      elif kw in ('figure', 'fig'):
        kwargs[kw] = self.figure
      elif kw in ('canvas',):
        kwargs[kw] = self.canvas
      elif kw in ('axes', 'ax'):
        kwargs[kw] = self.axes
    return kwargs

  def _begin_loop(self):
    if self.backend_is_TkAgg:
      self.window.after(self.Configs.show_after_ms, self.move_to_center)
    plt.show()

  # endregion: Private Methods

  # region: Public Methods

  def move_to(self, x:int, y:int):
    """Move figure's upper left corner to pixel (x, y)
    Reference: https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    """
    if self.backend_is_TkAgg:
      self.window.wm_geometry("+{}+{}".format(x, y))
    elif self.backend_is_WXAgg:
      self.window.SetPosition((x, y))
    else:
      # This works for QT and GTK, can also use window.setGeometry
      self.window.move(x, y)

  def move_to_center(self):
    """Should be called after window has been shown"""
    if self.backend_is_TkAgg:
      from lambo.gui.tkutils.misc import centerize_window
      centerize_window(self.window)
    else:
      pass

  def add_plotter(self, method, index=-1):
    assert callable(method)
    self.layer_plotter.insert(index, method)

  # endregion: Public Methods

  # region: Plotter Library

  @staticmethod
  def show_text(ax: plt.Axes, text="Yo, what's up."):
    ax.cla()
    ax.text(0.5, 0.5, text, ha='center', va='center')
    ax.set_axis_off()

  @staticmethod
  def imshow(x: np.ndarray, ax: plt.Axes = None, title=None, cmap=None,
             norm=None, aspect=None, interpolation=None, alpha=None,
             vmin=None, vmax=None):
    # Get current axes if ax is not provided
    if ax is None: ax = plt.gca()
    # Clear axes before drawing
    ax.cla()
    # Show images
    ax.imshow(x, cmap=cmap, norm=norm, aspect=aspect,
              interpolation=interpolation, alpha=alpha, vmin=vmin, vmax=vmax)

    # Set title if provided
    if title: ax.set_title(title)
    ax.set_axis_off()

  # endregion: Plotter Library


if __name__ == '__main__':
  b = Board()