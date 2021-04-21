from typing import Optional

import hashlib


def encrypt_md5(s: str, digit: Optional[int] = None) -> str:
  h = hashlib.md5()
  h.update(s.encode(encoding='utf-8'))
  code = h.hexdigest()
  if digit is not None:
    assert 0 < digit
    return code[:digit]
  return code


if __name__ == '__main__':
  print(encrypt_md5('0316'))
  print(encrypt_md5('1219'))
