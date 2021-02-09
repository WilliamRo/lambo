import matplotlib.pyplot as plt


def bind_quick_close():
  """Press Esc or q to close window after plt.show()"""
  def _on_key_press(event):
    if event.key in ('escape', 'q'):
      plt.close()
    else:
      print('>> key "{}" pressed'.format(event.key))

  plt.connect('key_press_event', _on_key_press)

