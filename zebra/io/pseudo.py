import os
import time

import numpy as np

from roma import console

from lambo.data_obj.interferogram import Interferogram
from lambo.analyzer.retriever import Retriever
from lambo.zebra.io.inflow import Inflow



class PseudoFetcher(Inflow):

  def __init__(self, path, seq_id=1, fps=10, max_len=20, L=None):
    super(PseudoFetcher, self).__init__(max_len)

    self.path = path
    self.seq_id = seq_id
    self.fetch_fps = fps

    self.interferograms = None
    self.index = 0

    self.L = L


  def _init(self):
    # Fetch all interferograms
    console.show_status(f'Reading interferograms from `{self.path}`')
    self.interferograms = Retriever.read_interferograms(
      self.path, seq_id=self.seq_id, radius=80)
    self.background = self.interferograms[0].bg_array
    self.interferograms = [ig.img for ig in self.interferograms]

    self._preprocess()

    assert self.fetch_fps > 0
    self.index = 0
    console.show_status('Looping ...')


  def _preprocess(self):
    if self.L is None: return
    console.show_status('Preprocessing ...')
    import cv2

    resized, dsize = [], (self.L, self.L)
    self.background = cv2.resize(self.background, dsize)
    for i, ig in enumerate(self.interferograms):
      resized.append(cv2.resize(ig, dsize))
      console.print_progress(i, len(self.interferograms))
    self.interferograms = resized


  def _loop(self):
    index = self.index % len(self.interferograms)
    self.append_to_buffer(self.interferograms[index])

    time.sleep(1 / self.fetch_fps * 0.9)
    self.index += 1



if __name__ == '__main__':
  trial_root = r'E:\lambai\01-PR\data'
  trial_names = ['01-3t3', '80-spacer-0526']
  path = os.path.join(trial_root, trial_names[1])

  pf = PseudoFetcher(path, fps=10)
  pf.open_gate()

  # Sleep for some time before end
  time.sleep(5)
