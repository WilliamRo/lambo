from roma import console
from roma import Nomear

import numpy as np
import re
import inspect


class Mind(Nomear):

  @property
  def command_history(self) -> list:
    return self.get_from_pocket('COMMAND_HISTORY', initializer=lambda: [])


  def sense(self):
    cmd = self.ask(history_buffer=self.command_history)
    if cmd is None: return
    cmd_string, func_key, args, kwargs = cmd

    # Add cmd_string to history buffer anyway
    self.command_history.insert(0, cmd_string)

    # Get method
    func = getattr(self, func_key, None)
    if not callable(func):
      self._err(' ! command `{}` not found.'.format(func_key))
      return

    # Try to execute func
    params_dict = inspect.signature(func).parameters
    params_values = list(params_dict.values())
    has_annotation = lambda p: p.annotation is not inspect._empty
    try:
      # Try to convert args type
      for i in range(len(args)):
        p = params_values[i]
        if has_annotation(p): args[i] = p.annotation(args[i])
      # Try to convert kwargs type
      for k, v in kwargs.items():
        p = params_dict[k]
        if has_annotation(p): kwargs[k] = p.annotation(v)

      # Execute
      func(*args, **kwargs)
    except Exception as e:
      self._err(' ! Failed to execute command `{}`'.format(cmd_string))
      self._err('.. Error Message:')
      self._err('- ' * 39)
      self._err(str(e))
      self._err('- ' * 39)


  @staticmethod
  def ask(history_buffer=()):
    from lambo.gui.tkutils.simple_dialogs import ask_string
    # Ask for command
    s = ask_string(history_buffer=history_buffer)
    if s is None: return None

    assert isinstance(s, str)
    parts = s.split(' ')
    assert len(parts) > 0

    # Parse string
    func_key, args, kwargs, flag = parts.pop(0), [], {}, True
    for p in parts:
      if '=' in p:
        _p = p.split('=')
        if len(_p) != 2:
          flag = False
          break
        kwargs[_p[0]] = _p[1]
      else: args.append(p)

    # Check and return
    if not flag:
      Mind._err(' ! `{}` is not an appropriate command'.format(s))
      return None
    return s, func_key, args, kwargs


  @staticmethod
  def _err(text):
    console.write_line(text, color='red')


if __name__ == '__main__':
  m = Mind()
  m.sense()
