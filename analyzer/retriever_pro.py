from typing import Callable, List, Optional

from lambo.analyzer.retriever import Retriever
from lambo.data_obj.interferogram import Interferogram

from roma import console

import numpy as np
import os
import time


class RetrieverPro(Retriever):

  def plot_extracted_unwrapped_phase(self, plot_3d=False):
    self._add_plot(lambda x: x.extracted_angle_unwrapped,
                   'Phase with Aberration', plot_3d=plot_3d)

  def plot_derivative_1_amp(self, plot_3d=False):
    def f(x: Interferogram):
      return np.log(np.abs(x.derivative_1_amp) + 1)
    self._add_plot(f, 'Gradient', plot_3d=plot_3d)

  def plot_aberration(self, plot_3d=False):
    self._add_plot(lambda x: x.aberration, 'Aberration', plot_3d=plot_3d)

  def plot_extracted_intensity(self):
    def f(x: Interferogram):
      return np.log(np.abs(x.extracted_image))
    self._add_plot(f, 'Extracted Intensity')


if __name__ == '__main__':
  trial_root = r'E:\lambai\01-PR\data'
  trial_name = '01-3t3'
  trial_name = '04-rbc'
  # trial_name = '05-bead'
  trial_name = '80-spacer-0526'
  path = os.path.join(trial_root, trial_name)

  radius = 80
  if trial_name == '80-spacer-0526': radius = 150
  r = RetrieverPro.initialize(path, radius)

  # r.save(overwrite=True, keys=['extracted_angle_unwrapped'])

  r.plot_interferogram()

  r.plot_extracted_phase()
  # r.plot_extracted_unwrapped_phase()
  r.plot_extracted_unwrapped_phase(True)
  # r.plot_aberration(True)
  # r.plot_extracted_unwrapped_phase()
  # r.plot_aberration()
  # r.plot_extracted_intensity()
  # r.plot_derivative_1_amp()
  r.plot_ground_truth()
  r.show()
