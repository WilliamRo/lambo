import tkinter as tk

from .misc import show_elegantly


def ask_string(history_buffer=()):
  """This widget works like pressing `:` in IdeaVim.

     history_buffer = [latest_cmd, ...]
  """
  # Insert current input into history buffer
  history_buffer = [''] + list(history_buffer)

  # box[0]: string to return
  # box[1]: history cursor
  # box[2]: allow to trigger text changed event
  box = [None, 0, True]

  root = tk.Tk()
  root.title('')
  root.resizable(0, 0)

  # Create text box and string var
  sv = tk.StringVar(root)
  def _text_modified():
    if box[-1]: history_buffer[0] = sv.get()
  sv.trace_add('write', lambda *args: _text_modified())
  text_box = tk.Entry(root, width=50, textvariable=sv)
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

    # Find cursor
    cursor = max(min((box[1] + d), len(history_buffer) - 1), 0)
    if cursor == box[1]: return

    # Update
    text_to_fill = history_buffer[cursor]
    box[1] = cursor

    # Fill (avoid triggering text changed event)
    box[-1] = False
    text_box.delete(0, tk.END)
    text_box.insert(0, text_to_fill)
    box[-1] = True

  root.bind('<Return>', lambda _: _close(True))
  root.bind('<Escape>', lambda _: _close(False))
  root.bind('<Control-n>', lambda _: _fill_in_history(-1))
  root.bind('<Control-p>', lambda _: _fill_in_history(1))

  # Display dialog at the center
  show_elegantly(root)

  # Return the text in the text box
  return box[0]

