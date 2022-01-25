import os, threading
import numpy as np

import matplotlib.pyplot as plt

from threading import Thread
from lambo.data_obj.interferogram import Interferogram
from lambo.gui.vinci.vinci import DaVinci
from lambo.zebra.io.inflow import Inflow
from lambo.zebra.io.exporter import Exporter
from lambo.zebra.decoder.decoder import Decoder
from lambo.zebra.gui.player import Player
from roma import console

from typing import Optional, List



class Zinci(DaVinci):

  def __init__(self, title='Zinci', height=5, width=5, max_fps=50):
    # Call parent's constructor
    super(Zinci, self).__init__(title, height, width)

    # Own attributes
    self.max_fps = max_fps
    self.fetcher: Optional[Inflow] = None
    self.decoders: List[Decoder] = []
    self.player: Optional[Player] = None
    self.main_thread = threading.current_thread()

    self.exporter = Exporter()

    self._register_keys()

  # region: Plotters

  def show_text(self, ax, text=None):
    if text is None:
      text = 'No data-fetcher found'
      if isinstance(self.fetcher, Inflow): text = 'Press [Space] to run'
    DaVinci.show_text(ax, text)

  def imshow_pro(
      self, x: np.ndarray, title=None, cmap=None, norm=None, aspect=None,
      interpolation=None, alpha=None):

    # # Save image if required
    if self.exporter.save_flag: self.exporter.save_image(x)

    self.imshow(x=x, ax=self.axes, title=title, norm=norm,
                aspect=aspect, interpolation=interpolation, alpha=alpha,
                vmin=self._color_limits[0], vmax=self._color_limits[1],
                color_bar=self._color_bar, k_space=self._k_space,
                log=self._k_space_log, cmap=self._cmap)
    self._zoom_in(ax=self.axes)

  # endregion: Plotters

  # region: Commands

  def _register_keys(self):
    self.state_machine.register_key_event(' ', self._play_or_pause)
    self.state_machine.register_key_event('enter', self._analyze_one)
    self.state_machine.register_key_event('p', self.exporter.set_path)

    self.state_machine.register_key_event('s', self.exporter.set_save_flag)

  # endregion: Commands

  # region: Public Methods

  def set_fetcher(self, fetcher: Inflow):
    assert isinstance(fetcher, Inflow)
    fetcher.master_thread = self.main_thread
    self.fetcher = fetcher
    self.state_machine.register_key_event('b', self.fetcher.set_background)

  def set_decoder(self, decoder: Decoder):
    assert isinstance(decoder, Decoder)
    decoder.plug(self)
    self.decoders.append(decoder)

  def display(self, auto_play=False):
    if auto_play and isinstance(self.fetcher, Inflow): self._play_or_pause()
    self.show()

  # endregion: Public Methods

  # region: Private Methods

  def _play_or_pause(self):
    # Initialize a player if necessary
    if self.player is None:
      self.show_text(self.axes, 'Opening gate ...')
      self.player = Player(self)
      self._internal_refresh()
      self.fetcher.open_gate()

    if self.player.is_playing: self.player.pause()
    else: self.player.play()

  def _analyze_one(self, radius=80):
    if not isinstance(self.fetcher, Inflow): return
    if len(self.fetcher.buffer) == 0: return
    if self.fetcher.background is None: return

    bg = self.fetcher.background
    img = self.fetcher.buffer[-1]
    Interferogram(img, bg, radius=radius).dashow(show_calibration=True)

  # endregion: Private Methods

  # region: Overwrite

  @DaVinci.layer_cursor.setter
  def layer_cursor(self, val):
    if len(self.layer_plotters) == 0: return
    previous_cursor = self.layer_cursor
    self._layer_cursor = val % len(self.layer_plotters)
    if self.layer_cursor != previous_cursor:
      # This is to avoid multi-threading conflicts
      if not self.player.is_playing: self._draw()

  # endregion: Overwrite



if __name__ == '__main__':
  from lambo.zebra.io.pseudo import PseudoFetcher
  from lambo.zebra.decoder.subtracter import Subtracter

  trial_root = r'E:\lambai\01-PR\data'
  trial_names = ['01-3t3', '80-spacer-0526']
  path = os.path.join(trial_root, trial_names[1])

  z = Zinci(height=6, width=6, max_fps=50)
  z.set_fetcher(PseudoFetcher(path, fps=20, L=1000, seq_id=4))
  z.set_decoder(Subtracter())
  z.set_decoder(Subtracter(boosted=True))

  z.display()