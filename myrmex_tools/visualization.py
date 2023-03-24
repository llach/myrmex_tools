import numpy as np

from .processing import _ensure_batch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def myrmex_to_rgb_image(data, cidx=2):
    """ converts a myrmex sample into and RGB image by filling one color channel with the force intensities
    """

    imgs = np.zeros(list(data.shape) + [3])
    imgs[:,:,:,cidx] = data
    imgs *= 255

    return np.squeeze(imgs.astype(np.uint8))

def upscale_repeat(frames, factor=10):
    """ upscaling for myrmex images by repeating the array N times; we avoid interpolation that would cause blurring of the sensor image.
    """
    frames = _ensure_batch(frames)
    return np.squeeze(frames.repeat(factor, axis=1).repeat(factor, axis=2))

def single_frame_heatmap(sample, fig, ax, with_colorbar=True):
    im = ax.imshow(sample, cmap="magma")

    # create colorbar
    if with_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

    # disabling  ticks makes things cleaner
    ax.set_xticks([])
    ax.set_yticks([])
    