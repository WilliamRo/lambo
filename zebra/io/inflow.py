import os, time
from threading import Thread
from typing import Optional

import numpy as np

from roma import Nomear
from roma import check_type, console

from lambo.zebra.base_classes.timer import Timer


class Inflow(Timer):
  """Base class"""

  def __init__(self, max_len=20):
    self.buffer: list = []
    self.background: Optional[np.ndarray] = None
    self.max_len = check_type(max_len, int)
    self.master_thread: Optional[Thread] = None
    self.thread: Optional[Thread] = None
    self.terminate_flag = False


  def set_background(self):
    if len(self.buffer) == 0: return
    self.background = self.buffer[-1]
    console.show_status('Background has been set')


  def open_gate(self):
    if isinstance(self.thread, Thread): return
      # print(' ! Gate has already been opened.')
      # return
    self.thread = Thread(target=self.fetch, name='DataFetcher')
    self.terminate_flag = False
    self.thread.start()


  def close_gate(self):
    if not isinstance(self.thread, Thread) or not self.thread.is_alive():
      raise AssertionError('!! No active thread found')
    self.terminate_flag = True


  def append_to_buffer(self, data):
    self.buffer.append(data)
    if len(self.buffer) > self.max_len: self.buffer.pop(0)


  def fetch(self):
    self._init()
    while True:
      if self.terminate_flag: break
      if (isinstance(self.master_thread, Thread)
          and not self.master_thread.is_alive()): break
      self._loop()
      self._tic()
    self._finalize()


  def _init(self): pass


  def _loop(self): raise NotImplemented


  def _finalize(self): pass



if __name__ == '__main__':
  pass
