import numpy as np

def _ensure_batch(d, data_dim=2):
    if type(d) == list: d = np.squeeze(d)
    if len(d.shape)==data_dim: d = np.expand_dims(d, axis=0)
    return d

def remove_outer(data, B=0):
    """ removes B outermost rows & columns of myrmex data; expects batch 
    """
    data = _ensure_batch(data)

    if B==0: return data
    assert B<=5, f"B<=5"
    return data[:,B:-B,B:-B]

def reshape_myrmex_vector(data):
    """ tactile data is published as a 1x256 vector in ROS, convert to 
    """
    data = _ensure_batch(data, data_dim=1)

    return np.reshape(data, list(data.shape[:-1])+[16,16])

def normalize_myrmex_data(data):
    """ raw myrmex readings are in [0,4095], convert to [0,1] with 1 = maximum force
    """
    return 1-(data/4095)

def filter_noise(data, noise_thresh=0.05):
    """ filter noise by setting all taxels measuring forces below a threshold to zero
    """
    if noise_thresh > 0.0: return np.where(data>noise_thresh, data, 10**-16) # don't use zero, otherwise we can have matrices with only zeros in them, posisbly breaking subsequent steps
    else: return data

def convert_tactile_message(data, B=0):
    """ wrapper for common preprocessing of myrmex data
    """
    return normalize_myrmex_data(remove_outer(reshape_myrmex_vector(data), B=B))
