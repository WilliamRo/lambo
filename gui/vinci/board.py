from matplotlib.gridspec import GridSpec
from matplotlib import cm

from roma import console, Nomear
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
    selected_rect = 'SELECTED_RECT'

  def __init__(self, title=None, height=5, width=5):
    # Set matploblib backend
    matplotlib.use('TkAgg')

    self.objects = []
    self.object_titles = []
    self.layer_plotters = []

    self._object_cursor = 0
    self._layer_cursor = 0

    # Attributes for bookmarks logic
    self.bookmarks = {}
    self._last_layer_cursor = 0
    self._last_obj_cursor = 0

    # Other attributes
    self.title = title
    self.fig_height = height
    self.fig_width = width

    # 2D options
    self._color_bar = False
    self._k_space = False
    self._k_space_log = False
    self._color_limits = [None, None]

    # 3D options
    self.keep_3D_view_angle = False
    self.view_angle = None
    self.z_lim_tuple: Optional[Tuple[float, float]] = None

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

    # Get arguments for current plotter
    kwargs = self._get_kwargs_for_plotter(self.current_plotter)

    # Plot
    self.current_plotter(**kwargs)

    # Set windows title
    self.canvas.set_window_title(self.win_title)

    # Set view angle if necessary
    if self.current_plotter_is_3D:
      if self.keep_3D_view_angle and self.view_angle is not None:
        self.axes3d.view_init(*self.view_angle)
      if self.z_lim_tuple is not None:
        assert len(self.z_lim_tuple) == 2
        self.axes3d.set_zlim3d(*self.z_lim_tuple)

    # Tight layout and refresh
    plt.tight_layout()

    # Refresh
    self._internal_refresh()

  def _clear(self):
    # Clear 2D axes
    if self.Keys.axes in self._cloud_pocket:
      self._cloud_pocket.pop(self.Keys.axes)

    # Clear 3D axes
    if self.Keys.axes3d in self._cloud_pocket:
      if self.keep_3D_view_angle:
        self.view_angle = (self.axes3d.elev, self.axes3d.azim)
      self._cloud_pocket.pop(self.Keys.axes3d)

    # Clear figure
    self.figure.clear()

  def _internal_refresh(self):
    self.canvas.draw()

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
      elif kw in ('k_space', 'k'):
        kwargs[kw] = self._k_space
      elif kw in ('color_bar', 'cb'):
        kwargs[kw] = self._color_bar
      elif kw in ('log', 'tl'):
        kwargs[kw] = self._k_space_log
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

  # region: Bookmark Logic

  def _determine_list(self, n):
    """1-5 for layers, 6-0 for objects"""
    assert isinstance(n, int) and 0 <= n <= 9
    if len(self.objects) <= 1 and len(self.layer_plotters) <= 1: return None
    if len(self.objects) <= 1: return self.layer_plotters
    if len(self.layer_plotters) <= 1: return self.objects
    return self.layer_plotters if 1 <= n <= 5 else self.objects

  def set_bookmark(self, n):
    assert isinstance(n, int) and 0 <= n <= 9
    channel = self._determine_list(n)
    if channel is None: return
    elif channel is self.layer_plotters:
      cursor, key = self.layer_cursor, 'plotter'
    else: cursor, key = self.object_cursor, 'object'

    self.bookmarks[n] = cursor
    console.show_status('Bookmark[{}] set on {}[{}]'.format(
      n, key, self.layer_cursor + 1))

  def jump_back_and_forth(self, n):
    if n not in self.bookmarks: return
    if self._determine_list(n) is self.layer_plotters:
      if self.layer_cursor != self.bookmarks[n]:
        self._last_layer_cursor = self.layer_cursor
        self.layer_cursor = self.bookmarks[n]
      else: self.layer_cursor = self._last_layer_cursor
    else:
      if self.object_cursor != self.bookmarks[n]:
        self._last_obj_cursor = self.object_cursor
        self.object_cursor = self.bookmarks[n]
      else: self.object_cursor = self._last_obj_cursor

  # endregion: Bookmark Logic

  def refresh(self): self._draw()

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

  def add_imshow_plotter(self, image, title=None, finalize=None):
    def plotter(ax: plt.Axes, color_bar: bool, k_space: bool, log: bool):
      self.imshow(
        image, ax, title=title, color_bar=color_bar, k_space=k_space, log=log)
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

  def imshow_pro(
      self, x: np.ndarray, title=None, cmap=None, norm=None, aspect=None,
      interpolation=None, alpha=None):
    self.imshow(x=x, ax=self.axes, title=title, cmap=cmap, norm=norm,
                aspect=aspect, interpolation=interpolation, alpha=alpha,
                vmin=self._color_limits[0], vmax=self._color_limits[1],
                color_bar=self._color_bar, k_space=self._k_space,
                log=self._k_space_log,
                rect=self.get_from_pocket(self.Keys.selected_rect))

  @staticmethod
  def imshow(x: np.ndarray, ax: plt.Axes = None, title=None, cmap=None,
             norm=None, aspect=None, interpolation=None, alpha=None,
             vmin=None, vmax=None, color_bar=False, k_space=False, log=False,
             rect=None):
    # Get current axes if ax is not provided
    if ax is None: ax = plt.gca()
    # Clear axes before drawing
    ax.cla()

    # Calculate spectrum density if necessary
    if k_space:
      x = np.abs(np.fft.fftshift(np.fft.fft2(x)))
      if log: x = np.log(x)

    # Show images
    im = ax.imshow(
      x, cmap=cmap, norm=norm, aspect=aspect, interpolation=interpolation,
      alpha=alpha, vmin=vmin, vmax=vmax)
    im.set_clim(vmin, vmax)

    # Set color bar if required
    if color_bar:
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      plt.colorbar(im, cax=cax)

    # Set title if provided
    if title: ax.set_title(title)
    ax.set_axis_off()

    # Zoom-in to rect if provided
    if rect is not None:
      Y, X = x.shape
      xlim, ylim = rect
      ax.set_xlim(*[l * X for l in xlim])
      ax.set_ylim(*[l * Y for l in ylim])

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

  def show_status(self, text: str):
    self.show_text(self.axes, text)
    self._internal_refresh()

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

  def set_clim(self, vmin: float = None, vmax: float = None):
    self._color_limits = [vmin, vmax]
    self._draw()
  clim = set_clim

  def toggle_freeze_zoom_in(self):
    """Works only for 2-D plot"""
    # Try to get rect info from pocket
    rect = self.get_from_pocket(self.Keys.selected_rect)
    # Get full size
    X, Y = [abs(se[1] - se[0]) for se in self.axes.images[0].sticky_edges[:2]]
    # Get relative x/y lim
    xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
    lims = ((xlim[0] / X, xlim[1] / X), (ylim[0] / Y, ylim[1] / Y))

    if rect == lims:
      self.replace_stuff(self.Keys.selected_rect, None)
      console.show_status('Freeze-zoom-in function has been turned off.')
    else:
      self.put_into_pocket(self.Keys.selected_rect, lims, False)
      console.show_status(
        'Zoom-in rect fixed to x:({:.1f}, {:.1f}), y:({:.1f}, {:.1f})'.format(
          *xlim, *ylim))
  fzi = toggle_freeze_zoom_in

  def export(self, which: str = None, fps: float = 2, cursor_range: str = None,
             fmt: str = 'gif', path: str = None, n_tail: int = 0):
    """If `fmt` is `mp4`, ffmpeg must be installed.
     Official instruction for windows system:
       https://www.wikihow.com/Install-FFmpeg-on-Windows
    """
    from roma import console
    import matplotlib.animation as animation
    import re

    # Set function
    if which in (None, '-', '*'):
      # Try to set _func automatically, not recommended
      which = 'o' if len(self.objects) > len(self.layer_plotters) else 'l'
    if which not in ('l', 'o'):
      raise ValueError('!! First parameter must be `o` or `l`')
    _func = self.slc if which == 'l' else self.soc

    # Find cursor range
    if cursor_range is None:
      begin, end = 1, len(self.layer_plotters if which == 'l' else self.objects)
    else:
      if re.match('^\d+:\d+$', cursor_range) is None:
        raise ValueError('!! Illegal cursor range `{}`'.format(cursor_range))
      begin, end = [int(n) for n in cursor_range.split(':')]

    end += 1

    # Find path
    if fmt not in ('gif', 'mp4'):
      raise KeyError('!! `fmt` should be `gif` or `mp4`')
    if path is None:
      from tkinter import filedialog
      path = filedialog.asksaveasfilename()
    if re.match('.*\.{}$'.format(fmt), path) is None:
      path += '.{}'.format(fmt)

    # Find movie writer
    writer = animation.FFMpegWriter(fps=fps)

    # Create animation
    tgt = 'objects' if which == 'o' else 'layers'
    frames = list(range(begin, end))

    # TODO: directly export mp4 file will lose last few frames. Use this code
    #       block to circumvent this issue temporarily
    if fmt == 'mp4' and n_tail > 0:
      frames.extend([frames[-1] for _ in range(n_tail)])

    console.show_status(
      'Saving animation ({}[{}:{}]) ...'.format(tgt, frames[0], frames[-1]))
    def func(n):
      console.print_progress(n - begin, total=end - begin)
      _func(n)

    # This line is important when this Board is not shown
    self._draw()

    ani = animation.FuncAnimation(
      self.figure, func, frames=frames, interval=1000 / fps)
    ani.save(path, writer=writer)
    console.show_status('Animation saved to `{}`.'.format(path))

  # endregion: Build-in Commands


if __name__ == '__main__':
  b = Board()