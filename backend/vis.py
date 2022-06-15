"""
Plotting and Visualisation
"""

__all__ = ['imshow_batch']

import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt


def imshow_batch(t: Tensor, n_width=8, *args, **kwargs):

    # Extract stensor shape information
    batch_size = t.shape[0]
    h, w = t.shape[-2:]

    # Calculate tiling
    n_width = batch_size if batch_size < n_width else n_width
    n_height = max(batch_size // n_width, 1)

    # Distribute Images
    img = np.zeros(shape=(n_height * h, n_width * w))
    img_idx = 0
    for i in range(n_width):
        for j in range(n_height):
            img[j * h:(j + 1) * h, i * w: (i + 1) * w] = t[img_idx].detach().cpu().numpy()
            img_idx += 1
            if img_idx >= batch_size:
                break

    return plt.imshow(img, *args, **kwargs)
