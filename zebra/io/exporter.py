import os, cv2

import numpy as np

from roma import Nomear
from roma import console



class Exporter(Nomear):

  def __init__(self):
    self.path = os.path.join(os.getcwd(), 'tmp')
    self.prefix = ''
    self.save_flag = False
    self._counter = 0

    console.show_status(f'Current path is `{self.path}`')


  @property
  def counter(self):
    self._counter += 1
    return self._counter


  def set_path(self):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory()

    if folder_path is None: return
    self.path = folder_path
    console.show_status(f'Path set to `{folder_path}`')


  def set_prefix(self, prefix=''):
    self.prefix = prefix
    console.show_status(f'Prefix set to `{self.prefix}`')


  def check_path(self):
    if not os.path.exists(self.path): os.mkdir(self.path)


  def set_save_flag(self): self.save_flag = True


  def save_image(self, image: np.ndarray, filp_flag=True):
    self.check_path()
    file_path = os.path.join(self.path, f'{self.prefix}{self.counter}.tif')

    # Check image
    if image.dtype in (np.float, ):
      image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imwrite(file_path, image)
    console.show_status(f'Image saved to `{file_path}`')
    if filp_flag: self.save_flag = False






