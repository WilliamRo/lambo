import matplotlib.pyplot as plt
from lambo.gui.vinci.events import StateMachine


def bind_quick_close():
  """Press Esc or q to close window after plt.show()"""
  StateMachine().bind_key_press_event()

