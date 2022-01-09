import os, time
import _thread, threading
import matplotlib.pyplot as plt

from typing import Optional
from threading import Thread

from roma import Nomear
from roma import console, check_type
from lambo.zebra.io.inflow import Inflow
from lambo.data_obj.interferogram import Interferogram



class Player(Nomear):

  MAX_TICS = 5

  def __init__(self, zinci):
    # work-around for friend class
    from lambo.zebra.gui.zinci import Zinci

    self.zinci: Zinci = zinci
    self.zinci.add_plotter(self.zinci.imshow_pro)

    # Register decoder if provider
    for d in self.zinci.decoders: self.zinci.add_plotter(d.plotter)

    self._fringe = None
    self._loop_flag = False

    # Timers
    self._inflow_tics = []

  # region: Properties

  @property
  def main_thread(self) -> Thread: return self.zinci.main_thread

  @property
  def inflow(self) -> Inflow: return self.zinci.fetcher

  @property
  def is_playing(self) -> bool: return self._loop_flag

  @property
  def max_fps(self) -> Optional[int]: return self.zinci.max_fps

  @property
  def inflow_fps(self) -> float:
    L = len(self._inflow_tics)
    if L < 2: return 0
    return (L - 1) / (self._inflow_tics[-1] - self._inflow_tics[0])

  @property
  def duration(self) -> Optional[float]:
    if len(self._inflow_tics) < 2: return None
    return self._inflow_tics[-1] - self._inflow_tics[-2]

  # endregion: Properties

  # region: Private Methods

  def _tic(self, inflow=True):
    assert inflow
    self._inflow_tics.append(time.time())
    if len(self._inflow_tics) > self.MAX_TICS: self._inflow_tics.pop(0)

  def _check_idle(self):
    if not isinstance(self.inflow, Inflow): return True
    if len(self.inflow.buffer) == 0: return True
    fringe = self.inflow.buffer[-1]
    if isinstance(fringe, Interferogram): fringe = fringe.img
    if self._fringe is fringe: return True

    # Update latest fringe
    self._fringe = fringe
    return False

  def _safe_refresh(self): self.zinci.canvas.draw_idle()

  def _show_fringe(self):
    self.zinci.imshow(self._fringe, self.zinci.axes)
    self._safe_refresh()

  def _loop(self):
    self._tic()
    while True:
      # Termination check
      if not all([self.main_thread.is_alive(), self._loop_flag]): return
      # Check idle
      if self._check_idle(): continue

      # Plot
      self.zinci.objects = [self._fringe]
      self.zinci.object_titles = [f'Player FPS: {self.inflow_fps:.1f}']
      self.zinci.refresh(in_thread=True)
      self._tic(inflow=True)

      # Sleep to maintain the max_fps
      if self.max_fps is not None:
        time.sleep(1 / self.max_fps)
        # plt.pause(0.0001)  # time.sleep is better than plt.pause

  # endregion: Private Methods

  # region: Public Methods

  def play(self):
    if self._loop_flag is True: return
    _thread.start_new_thread(self._loop, ())
    self._loop_flag = True

  def pause(self):
    if self._loop_flag is False: return
    self._loop_flag = False
    self._inflow_tics = []
    console.show_status('Pause')

  # endregion: Public Methods


