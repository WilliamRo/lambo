#


def centerize_window(window):
  h, w = window.winfo_height(), window.winfo_width()
  H, W = window.winfo_screenheight(), window.winfo_screenwidth()
  x, y = (W - w) // 2, (H - h) // 2
  window.geometry("+{}+{}".format(x, y))


def show_elegantly(window):
  window.focus_force()
  window.after(1, lambda: centerize_window(window))
  window.mainloop()

