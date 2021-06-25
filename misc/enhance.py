from collections.abc import Iterable


def descartes(*seqs):
  # Sanity check
  assert len(seqs) > 0
  for s in seqs: assert isinstance(s, Iterable)

  seqs = list(seqs)
  first_seq = seqs.pop(0)

  for obj in first_seq:
    if len(seqs) == 0: yield (obj,)
    else:
      for tail in descartes(*seqs):
        yield (obj,) + tail


if __name__ == '__main__':
  for i, j, c in descartes(range(3), (3, 4), 'abc'):
    print(i, j, c)