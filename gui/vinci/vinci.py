from lambo.gui.vinci.board import Board
from lambo.gui.vinci.events import StateMachine
from lambo.gui.vinci.mind import Mind


class DaVinci(Board, Mind):

  def __init__(self, title=None, height=5, width=5, init_as_image_viewer=False):
    # Call parent's constructor
    super(DaVinci, self).__init__(title, height, width)

    # Attributes
    self.state_machine = StateMachine()

    # Finalize the initiation
    self._register_events_for_cursors()
    self._register_events_for_switches()

    self._activate_mind()
    if init_as_image_viewer: self.add_plotter(self.imshow_pro)

  # region: Properties

  # endregion: Properties

  # region: Private Methods

  def _toggle_color_bar(self):
    self._color_bar = not self._color_bar
    self._draw()

  def _toggle_display_spectrum(self):
    self._k_space = not self._k_space
    self._draw()

  def _register_events_for_switches(self):
    self.state_machine.register_key_event('C', self._toggle_color_bar)
    self.state_machine.register_key_event('F', self._toggle_display_spectrum)

  def _register_events_for_cursors(self):

    def _move_cursor(obj_cursor_shift: int, layer_cursor_shift: int):
      assert obj_cursor_shift == 0 or layer_cursor_shift == 0
      self.object_cursor += obj_cursor_shift
      self.layer_cursor += layer_cursor_shift

    obj_forward = lambda: _move_cursor(1, 0)
    self.state_machine.register_key_event('j', obj_forward)
    self.state_machine.register_key_event('right', obj_forward)

    obj_backward = lambda: _move_cursor(-1, 0)
    self.state_machine.register_key_event('k', obj_backward)
    self.state_machine.register_key_event('left', obj_backward)

    layer_forward = lambda: _move_cursor(0, 1)
    self.state_machine.register_key_event('l', layer_forward)
    self.state_machine.register_key_event('down', layer_forward)

    layer_backward = lambda: _move_cursor(0, -1)
    self.state_machine.register_key_event('h', layer_backward)
    self.state_machine.register_key_event('up', layer_backward)

    # Register events for bookmark
    for n in range(1, 10):
      self.state_machine.register_key_event(
        'ctrl+{}'.format(n), lambda n=n: self.set_bookmark(n))
      self.state_machine.register_key_event(
        str(n), lambda n=n: self.jump_back_and_forth(n))

  def _activate_mind(self):
    if self.backend_is_TkAgg:
      self.state_machine.register_key_event(':', self.sense)

  # endregion: Private Methods

  # region: Build-in Commands

  def set_z_lim(self, zmin: float, zmax: float):
    assert zmin < zmax
    self.z_lim_tuple = (zmin, zmax)
    self._draw()
  zlim = set_z_lim

  def toggle_log(self):
    self._k_space_log = not self._k_space_log
    self._draw()
  log = tl = toggle_log

  # endregion: Build-in Commands

  # region: Public Methods

  def show(self):
    self._draw()
    self.state_machine.bind_key_press_event(board=self)
    self._begin_loop()

  # endregion: Public Methods

  # region: Public Static Methods

  @staticmethod
  def draw(images: list, titles=None, flatten=False):
    assert not flatten
    da = DaVinci()
    da.objects = images
    da.object_titles = [] if titles is None else titles
    da.add_plotter(da.imshow)
    da.show()

  # endregion: Public Static Methods


if __name__ == '__main__':
  dv = DaVinci('DaVinci').show()
