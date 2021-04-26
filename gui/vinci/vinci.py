from lambo.gui.vinci.board import Board
from lambo.gui.vinci.events import StateMachine


class DaVinci(Board):

  def __init__(self, title=None, height=5, width=5):
    # Call parent's constructor
    super(DaVinci, self).__init__(title, height, width)

    # Attributes
    self.state_machine = StateMachine()

    # Finalize the initiation
    self._register_events_for_cursors()

  # region: Properties

  # endregion: Properties

  # region: Private Methods

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

  def _register_admin(self):
    if self.backend_is_TkAgg:
      pass

  # endregion: Private Methods

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
    dv = DaVinci()
    dv.objects = images
    dv.add_plotter(dv.imshow)
    dv.show()

  # endregion: Public Static Methods


if __name__ == '__main__':
  dv = DaVinci('DaVinci').show()
