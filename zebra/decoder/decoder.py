import time

import numpy as np

from roma import Nomear
from typing import Optional



class Decoder(Nomear):

  def __init__(self):
    # work-around for friend class
    from lambo.zebra.gui.zinci import Zinci

    self.zinci: Optional[Zinci] = None

    self.durations = []
    self._tic_stamp = None


  def tic(self): self._tic_stamp = time.time()


  def toc(self):
    MAX_LEN = 10
    assert self._tic_stamp is not None
    self.durations.append(time.time() - self._tic_stamp)
    if len(self.durations) > MAX_LEN: self.durations.pop(0)


  def plotter(self, x, title=None):
    # Process
    self.tic()
    x = self._decode(x)
    self.toc()

    # Add to title
    if title is None: title = ''
    fps = 1.0 / np.average(self.durations)
    title += f' | Decoder FPS: {fps:.1f}'

    # Show result
    self.zinci.imshow_pro(x, title)


  def plug(self, zinci): self.zinci = zinci


  def _decode(self, x):
    return x

