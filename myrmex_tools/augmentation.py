import torch
import numpy as np
import torch.nn.functional as F

from numpy.random import randint, choice

def get_pad_and_slice(shift):
         if shift <= 0:
            return [-shift, 0], slice(0, shift)
         else:
            return [0, shift], slice(shift, 1000)

def shift_columns(frames, rpad, rsli, cpad, csli):
    """ 

    frames.shape = [16,16] = [H,W] (single frame)
    OR 
    frames.shape = [N,16,16] = [batch,H,W] (sequence)

    pad: how many columns we need to pad
    shift: how many columns we'll shift

    padding list: [left, right, top, bottom, front, back]
    -> indices 0,1 are columns, indices 2,3 are rows
    """
    frames = torch.Tensor(frames)
    return F.pad(frames, pad=cpad+rpad+[0,0])[:,:,rsli,csli].numpy()

def random_shift_seq(seq, augment):
    rows, cols = augment
    if rows: 
        rpad, rsli = get_pad_and_slice(choice([randint(-4,0), randint(1,5)]))
    else:
        rpad, rsli = [0,0], slice(-100000, 100000)
    if cols: 
        cpad, csli = get_pad_and_slice(choice([randint(-4,0), randint(1,5)]))
    else:
        cpad, csli = [0,0], slice(-100000, 100000)
    return np.squeeze(shift_columns(seq, rpad=rpad, rsli=rsli, cpad=cpad, csli=csli))