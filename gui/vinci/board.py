from matplotlib.gridspec import GridSpec
from matplotlib import cm

from lambo.abstract.noear import Nomear
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional

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
    axes3d = 'AXES3D'

  def __init__(self, title=None, height=5, width=5):
    self.objects = []
    self.object_titles = []
    self.layer_plotters = []

    self._object_cursor = 0
    self._layer_cursor = 0

    # Other attributes
    self.title = title
    self.fig_height = height
    self.fig_width = width

    # 3D options
    self.keep_3D_view_angle = False
    self.view_angle = None
    self.z_lim: Optional[Tuple[float, float]] = None

  # region: Properties

  @property
  def object_cursor(self):
    return self._object_cursor

  @object_cursor.setter
  def object_cursor(self, val):
    if len(self.objects) == 0: return
    previous_cursor = self.object_cursor
    self._object_cursor = val % len(self.objects)
    if self.object_cursor != previous_cursor: self._draw()

  @property
  def layer_cursor(self):
    return self._layer_cursor

  @layer_cursor.setter
  def layer_cursor(self, val):
    if len(self.layer_plotters) == 0: return
    previous_cursor = self.layer_cursor
    self._layer_cursor = val % len(self.layer_plotters)
    if self.layer_cursor != previous_cursor: self._draw()

  @property
  def current_plotter(self):
    if not self.layer_plotters: return self.show_text
    return self.layer_plotters[self.layer_cursor]

  @property
  def current_plotter_is_3D(self):
    # Get method signature
    sig = inspect.signature(self.current_plotter).parameters
    return any([key in sig.keys() for key in ('axes3d', 'ax3d')])

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
  def axes3d(self) -> Axes3D:
    init_axes = lambda: self.figure.add_subplot(111, projection='3d')
    return self.get_from_pocket(self.Keys.axes3d, initializer=init_axes)

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
    if len(self.objects) > 1:
      result = '[{}/{}]'.format(self.object_cursor + 1, len(self.objects))
    if len(self.layer_plotters) > 1:
      result += '[{}/{}]'.format(
        self.layer_cursor + 1, len(self.layer_plotters))

    if self.title is not None: result = ' '.join([result, self.title])
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

    # Set view angle if necessary
    if self.current_plotter_is_3D:
      if self.keep_3D_view_angle and self.view_angle is not None:
        self.axes3d.view_init(*self.view_angle)
      if self.z_lim is not None:
        assert len(self.z_lim) == 2
        self.axes3d.set_zlim3d(*self.z_lim)

    # Tight layout and refresh
    plt.tight_layout()

    # Refresh
    self.canvas.draw()

  def _clear(self):
    # Clear 2D axes
    if self.Keys.axes in self._pocket:
      self._pocket.pop(self.Keys.axes)

    # Clear 3D axes
    if self.Keys.axes3d in self._pocket:
      if self.keep_3D_view_angle:
        self.view_angle = (self.axes3d.elev, self.axes3d.azim)
      self._pocket.pop(self.Keys.axes3d)

    # Clear figure
    self.figure.clear()

  def _get_kwargs_for_plotter(self, plotter):
    assert callable(plotter)
    # Get method signature
    sig = inspect.signature(plotter).parameters
    # Get key-word arguments for method according to its signature
    kwargs = {}
    for kw in sig.keys():
      if kw in ('obj', 'x', 'img', 'im'):
        kwargs[kw] = self.objects[self.object_cursor]
      elif kw in ('figure', 'fig'):
        kwargs[kw] = self.figure
      elif kw in ('canvas',):
        kwargs[kw] = self.canvas
      elif kw in ('axes', 'ax'):
        kwargs[kw] = self.axes
      elif kw in ('axes3d', 'ax3d'):
        kwargs[kw] = self.axes3d
      elif kw in ('title', 'im_title'):
        if len(self.object_titles) == len(self.objects):
          kwargs[kw] = self.object_titles[self.object_cursor]
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
    if index == -1: index = len(self.layer_plotters)
    self.layer_plotters.insert(index, method)

  def add_imshow_plotter(
      self, image, title=None, finalize=None, color_bar=False):
    def plotter(ax: plt.Axes):
      self.imshow(image, ax, title=title, color_bar=color_bar)
      if callable(finalize): finalize()
    self.add_plotter(plotter)

  def add_image(self, im, title=None):
    self.objects.append(im)
    if title is not None: self.object_titles.append(title)

  # endregion: Public Methods

  # region: Plotter Library

  # region: Toolbox

  def set_im_axes(self, ax: plt.Axes = None, title=None):
    if ax is None: ax = self.axes
    ax.set_axis_off()
    if title: ax.set_title(title)

  # endregion: Toolbox

  @staticmethod
  def histogram(x: np.ndarray, ax: plt.Axes, bins=20, density=True):
    pixels = np.ravel(x)
    ax.hist(pixels, bins=bins, density=density)
    ax.set_aspect('auto')

  @staticmethod
  def show_text(ax: plt.Axes, text="Yo, what's up."):
    ax.cla()
    ax.text(0.5, 0.5, text, ha='center', va='center')
    ax.set_axis_off()

  @staticmethod
  def imshow(x: np.ndarray, ax: plt.Axes = None, title=None, cmap=None,
             norm=None, aspect=None, interpolation=None, alpha=None,
             vmin=None, vmax=None, color_bar=False):
    # Get current axes if ax is not provided
    if ax is None: ax = plt.gca()
    # Clear axes before drawing
    ax.cla()
    # Show images
    im = ax.imshow(
      x, cmap=cmap, norm=norm, aspect=aspect, interpolation=interpolation,
      alpha=alpha, vmin=vmin, vmax=vmax)

    # Set color bar if required
    if color_bar:
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      plt.colorbar(im, cax=cax)

    # Set title if provided
    if title: ax.set_title(title)
    ax.set_axis_off()

  @staticmethod
  def scatter(X, Y, Z, ax3d: Axes3D, **kwargs):
    ax3d.scatter(X, Y, Z, c=Z, **kwargs)

  @staticmethod
  def plot_im_as_3d(im: np.ndarray, func, **kwargs) -> plt.Axes:
    H, W = im.shape
    X, Y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return func(X, Y, im, **kwargs)

  @staticmethod
  def plot3d(im: np.ndarray, ax3d: Axes3D, cmap=cm.coolwarm, title=None,
             **kwargs):
    r = Board.plot_im_as_3d(im, ax3d.plot_surface, cmap=cmap, **kwargs)
    if title is not None: ax3d.set_title(title)
    return r

  @staticmethod
  def plot_wireframe(im: np.ndarray, ax3d: Axes3D, color='green', **kwargs):
    return Board.plot_im_as_3d(im, ax3d.plot_wireframe, color=color, **kwargs)

  # endregion: Plotter Library

  # region: Build-in Commands

  def poi(self):
    """Print Object Information"""
    if not self.objects:
      print(' ! not object found')
      return
    data = self.objects[self.object_cursor]
    assert isinstance(data, np.ndarray)
    print('Information of object[{}]:'.format(self.object_cursor + 1))
    print('.. Shape = {}'.format(data.shape))
    print('.. Range = [{:.3f}, {:.3f}]'.format(np.min(data), np.max(data)))

  def slc(self, n: int): self.layer_cursor = n - 1
  def soc(self, n: int): self.object_cursor = n - 1

  # endregion: Build-in Commands


if __name__ == '__main__':
  b = Board()