import tkinter as tk

from .misc import show_elegantly


def ask_string():
  box = [None]
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


  root.bind('<Return>', lambda _: _close(True))
  root.bind('<Escape>', lambda _: _close(False))

  show_elegantly(root)

  # Return the text in the text box
  return box[0]

