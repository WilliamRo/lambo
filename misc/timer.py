from collections import OrderedDict
import time


class Timer(object):

  def __init__(self, verbose: bool = False, unit: str = 'ms'):
    self._verbose: bool = verbose
    self._unit = unit

    self._tics: OrderedDict = OrderedDict()
    self._records: OrderedDict = OrderedDict()

  def tic_toc(self, key):
    return TicToc(key, self)

  def tic(self, key):
    assert key not in self._tics
    self._tics[key] = time.time()

  def toc(self, key=None):
    # Get tic
    assert len(self._tics) > 0
    if key is None: key = list(self._tics.keys())[-1]
    elapsed = time.time() - self._tics.pop(key)

    # Apply unit
    if self._unit in ('ms', 'millisecond'): elapsed *= 1000
    else: assert self._unit in ('sec', 'second')

    if self._verbose: self._print_time(key, elapsed)
    if key in self._records: self._records[key] += elapsed
    else: self._records[key] = elapsed

  def report(self):
    total = sum(self._records.values())
    for k, v in self._records.items():
      self._print_time(k, v, ' {:.1f}%'.format(v / total * 100))
    self._print_time('ALL', total)

  def _print_time(self, k, v, tail=''):
    msg = 'Elapsed time for {} is {:.2f}{}'.format(k, v, self._unit)
    print(msg + tail)


class TicToc(object):

  def __init__(self, key, timer: Timer):
    self.key = key
    self.timer: Timer = timer

  def __enter__(self):
    self.timer.tic(self.key)

  def __exit__(self, *args, **kwargs):
    self.timer.toc(self.key)


if __name__ == '__main__':
  tm = Timer()
  for i in range(1, 6):
    with tm.tic_toc('Stage {}'.format(i)):
      time.sleep(0.1 * i)
  tm.report()



