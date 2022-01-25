import time

from roma import Nomear



class Timer(Nomear):

  MAX_TICS = 5


  @Nomear.property(key='tics')
  def tic_dict(self):
    return {}


  @property
  def fps(self, key='default') -> float:
    if key not in self.tic_dict: self.tic_dict[key] = []
    tics = self.tic_dict[key]

    L = len(tics)
    if L < 2: return 0
    return (L - 1) / (tics[-1] - tics[0])


  def _tic(self, key='default'):
    if key not in self.tic_dict: self.tic_dict[key] = []
    tics = self.tic_dict[key]

    tics.append(time.time())
    if len(tics) > self.MAX_TICS: tics.pop(0)


  def _reset_tics(self, key='default'):
    self.tic_dict[key].clear()
