import numpy as np

from numpy.random import randint, choice
from myrmex_tools.processing import _ensure_batch

def _rotate(samples, k):
    """ rotates a single sample or batch by k*90 degrees
    """
    samples = _ensure_batch(samples)
    return np.squeeze(np.rot90(samples, k=k, axes=(1,2)))

def _noise(samples, sd):
    """ adds random noise to a single sample or batch, clipping values at 0
    """
    samples = _ensure_batch(samples)
    return np.squeeze(np.clip(samples+np.random.normal(loc=0, scale=sd, size=samples.shape), 0, 1))

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
    return _rotate(samples, k=np.random.randint(0, 4))

def full_augment(sample, nnoise=4, sd=0.01): # TODO determine correct sd
    """
    in:  (N,16,16)
    out: (M,16,16)

    deterministic augmentation of sensor samples.
    """
    samples = []
    for rot_sample in [_rotate(sample, k=k) for k in range(4)]:
        for flip_sample in [rot_sample, np.flip(rot_sample, axis=1)]:
            for noise_sample in [flip_sample]+[_noise(flip_sample, sd=sd) for _ in range(nnoise)]:
                samples.append(noise_sample)
    return np.array(samples)