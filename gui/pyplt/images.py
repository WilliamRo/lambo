import matplotlib.pyplot as plt

from .events import bind_quick_close


def imshow(*imgs, max_cols=3, zoom_coef=1.0):
  # Get image number
  N = len(imgs)
  assert N > 0

  # Determine subplot layout
  nrows = N // max_cols + 1
  ncols = min(max_cols, N)
  plt.figure(1, figsize=[zoom_coef * 4.0 * s for s in (ncols, nrows)])

  # Tile images
  for i, img in enumerate(imgs):
    plt.subplot(nrows, ncols, i + 1)
    plt.axis('off')
    plt.imshow(img)

  # Bind quick close shortcut to current figure
  bind_quick_close()

  # Make the figure look nicer
  plt.tight_layout()

  # Show images
  plt.show()
