from lambo.data_obj.interferogram import Interferogram
from lambo.zebra.decoder.decoder import Decoder

import numpy as np



class Subtracter(Decoder):

  def __init__(self, radius=80, boosted=False):
    # Call parent's constructor
    super(Subtracter, self).__init__()

    # Customized variables
    self.radius = radius
    self.boosted = boosted

  # region: Properties

  @Decoder.property(key='background')
  def background(self) -> Interferogram:
    bg = Interferogram(self.zinci.fetcher.background, radius=self.radius)
    if self.boosted: bg.booster = True
    return bg

  # endregion: Properties

  # region: Core Methods

  def _decode(self, x):
    ig = Interferogram(
      x, radius=self.radius, peak_index=self.background.peak_index)
    ig._backgrounds = [self.background]
    if self.boosted: ig.booster = True
    y = ig.unwrapped_phase
    ig.release()
    return y

  # endregion: Core Methods

  # region: Other Methods

  def plug(self, zinci):
    super(Subtracter, self).plug(zinci)
    def _set_background():
      self.zinci.fetcher.set_background()
      self.get_from_pocket('background', default=None, put_back=False)
    self.zinci.state_machine.library['b'] = _set_background

  # endregion: Other Methods
