import tkinter as tk

from .misc import show_elegantly


def ask_string(history_buffer=()):
  box = [None, None]
  root = tk.Tk()
  root.title('')
  root.resizable(0, 0)

  text_box = tk.Entry(root, width=50)
  text_box.pack()
  text_box.focus()


  # Bind events
  def _close(coin: bool):
    if coin: box[0] = text_box.get()
    root.destroy()
    root.quit()

  def _fill_in_history(d):
    if not history_buffer: return
    assert d in (-1, 1)
    if box[1] is None: c = len(history_buffer) - 1
    else: c = (box[1] + d) % len(history_buffer)
    # Fill
    text_box.delete(0, tk.END)
    text_box.insert(0, history_buffer[c])
    # Update
    box[1] = c

  root.bind('<Return>', lambda _: _close(True))
  root.bind('<Escape>', lambda _: _close(False))
  root.bind('<Control-n>', lambda _: _fill_in_history(1))
  root.bind('<Control-p>', lambda _: _fill_in_history(-1))

  show_elegantly(root)

  # Return the text in the text box
  return box[0]

