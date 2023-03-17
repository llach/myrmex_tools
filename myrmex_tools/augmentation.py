import numpy as np

from numpy.random import randint, choice
from myrmex_tools.processing import _ensure_batch


def get_pad_and_slice(shift):
    """ returns padding and slice for random sensor image translation
    """
    if shift <= 0:
        return [-shift, 0], slice(0, shift)
    else:
        return [0, shift], slice(shift, None)

def random_translation(samples, augment_dims = [True,True]):
    """ applies a random translation to a batch of samples (N,16,16).
    """

    # flags to either augment row-wise, column-wise or both
    if not np.any(augment_dims): return samples
    rows, cols = augment_dims

    # get padding & slice for each dimension that should be augmented 
    row_pad, row_slice = get_pad_and_slice(choice([randint(-4,0), randint(1,5)])) if rows else ([0,0], slice(0, None))
    column_pad, column_slice = get_pad_and_slice(choice([randint(-4,0), randint(1,5)])) if cols else ([0,0], slice(0, None))

    # ensure we have a batch dimension, then pad and slice the sample sequence
    samples = _ensure_batch(samples)
    return np.pad(samples, [[0,0], row_pad, column_pad])[:,row_slice,column_slice]

def random_rotate90(samples):
    """ rotates a batch of myrmex samples randomly by 90 degrees
    !!! IMPORTANT !!! this does not work if left and right myrmex image are stacked in the batch dimension and k is uneven, since one side is flipped.
    """
    samples = _ensure_batch(samples)
    return np.rot90(samples, k=np.random.randint(0, 4), axes=(1,2))