import numpy as np
import matplotlib.pyplot as plt

from typing import Union
from .events import bind_quick_close


def imshow(*imgs, titles: Union[None, list, tuple] = None,
           max_cols=3, zoom_coef=1.0):
  # Get image number
  N = len(imgs)
  assert N > 0
  if titles: assert len(titles) == N

  # Determine subplot layout
  ncols = min(max_cols, N)
  nrows = int(np.ceil(N / ncols))
  plt.figure(1, figsize=[zoom_coef * 4.0 * s for s in (ncols, nrows)])

  # Tile images
  for i, img in enumerate(imgs):
    decorate = None
    if isinstance(img, (list, tuple)): img, decorate = img
    plt.subplot(nrows, ncols, i + 1)
    plt.axis('off')
    plt.imshow(img)
    if titles: plt.title(titles[i])
    if callable(decorate): decorate()

  # Bind quick close shortcut to current figure
  bind_quick_close()

  # Make the figure look nicer
  plt.tight_layout()

  # Show images
  plt.show()
