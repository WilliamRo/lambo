from typing import Callable, List, Optional

from lambo.analyzer.retriever import Retriever
from lambo.data_obj.interferogram import Interferogram

from roma import console

import numpy as np
import os
import time


class RetrieverPro(Retriever):

  def plot_interferogram(self, R=None):
    self._add_plot(lambda x: self.crop(x.img, R), 'Interferogram')

  def plot_ground_truth_2d(self, R=None):
    self._add_plot(lambda x: self.crop(x.flattened_phase, R), 'Ground Truth')

  def plot_extracted_image(self, part=None, R=None, plot_3d=False):
    f = {None: np.abs, 'r': np.real, 'i': np.imag}
    self._add_plot(
      lambda x: f[part](self.crop(x.extracted_image, R)),
      'Extracted Image ({})'.format(
        {None: 'magnitude', 'r': 'real', 'i': 'image'}[part]), plot_3d=plot_3d)

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

  def plot_local_match(self, kernel_size, fmt='abs', R=None, level=1,
                       k_space=False, round_kernel=False, angle=0, r=1.0):
    from scipy import signal

    def f(x: Interferogram):
      kernel = x.get_fourier_prior(
        kernel_size, fmt='complex', rotundity=round_kernel, angle=angle, r=r)
      y = self.crop(x.img, R)
      for lv in range(level):
        y = self.get_from_pocket(
          'local_{}[{}][lv{}]'.format(kernel, self.object_cursor, lv),
          initializer=lambda: signal.convolve2d(y, kernel, mode='same'))
        add = lambda v: np.abs(np.real(v)) + np.abs(np.imag(v))
        y = {'r': np.real, None: np.abs, 'i': np.imag, 'add': add,
             'abs': np.abs}[fmt](y)

      if k_space: y = np.log(np.abs(np.fft.fftshift(np.fft.fft2(y))))
      return y

    title = '{}(Local Match)-lv{}-a{}-r{:.2f}-ks{}]'.format(
      fmt, level, angle, r, kernel_size)
    if round_kernel: title += '(round kernel)'
    if k_space: title += ' in K-Space'
    self._add_plot(f, title)


if __name__ == '__main__':
  trial_root = r'E:\lambai\01-PR\data'
  trial_name = '01-3t3'
  trial_name = '04-rbc'
  # trial_name = '05-bead'
  # trial_name = '80-spacer-0526'
  path = os.path.join(trial_root, trial_name)

  radius = 80
  if trial_name == '80-spacer-0526': radius = 150
  r = RetrieverPro.initialize(path, radius)

  # r.plot_extracted_image()
  crop = 250

  rs = (0.5, )
  rs = np.linspace(0.01, 0.99, 50)
  Ls = (25,)
  # Ls = list(range(3, 50, 2))
  r.show_dual_conv(
    Ls=Ls,
    omega=999,
    rs=rs,
    crop=250,
    rotundity=True,
    show_sum=False,
    show_real_imag=False,
    show_extracted=False,
    include_origin=False,
    # combinations=(((0, 30), (0.6,)),),
  )

  # R = crop
  # kernel_sizes = (9, 17, 31)
  # # kernel_sizes = (21,)
  # angles = (-5, 0, 5)
  #
  # # r.save(overwrite=True, keys=['extracted_angle_unwrapped'])
  #
  # r.plot_interferogram(R=R)
  #
  # r.plot_extracted_image('r', R=R, plot_3d=False)
  # r.plot_extracted_image('i', R=R)
  #
  # r.plot_extracted_image(None, R=R)
  #
  # for ks in kernel_sizes:
  #   for a in angles:
  #     # r.plot_local_match(ks, None, R=R, angle=a)
  #     r.plot_local_match(ks, None, R=R, round_kernel=True, angle=a, r=1.0)
  #     # r.plot_local_match(ks, None, R=R, round_kernel=True, angle=a, r=1.1)
  #     # r.plot_local_match(ks, None, R=R, round_kernel=True, angle=a, r=0.9)
  #   # r.plot_local_match(ks, None, R=R, level=2)
  #   #   r.plot_local_match(ks, 'r', R=R)
  #   #   r.plot_local_match(ks, 'i', R=R)
  #
  # # r.plot_extracted_phase()
  # # r.plot_extracted_unwrapped_phase()
  # # r.plot_extracted_unwrapped_phase(False)
  # # r.plot_aberration(True)
  # # r.plot_extracted_unwrapped_phase()
  # # r.plot_aberration()
  # # r.plot_extracted_intensity()
  # # r.plot_derivative_1_amp()
  # r.plot_ground_truth_2d(R)
  # r.show()
