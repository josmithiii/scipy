import numpy as np

# Return a complete spectrum given the non-negative-frequency portion,
# including dc and fs/2:
def append_flip_conjugate(X, is_magnitude=False):
    """
    Append the flipped conjugate of the input array, excluding the
    first and last elements (normally dc and fs/2) in what is flipped and appended.

    Parameters:
    X (np.ndarray): Input array, typically a spectrum over [0,pi], inclusive.

    Returns:
    np.ndarray: Array with appended flipped conjugate interior, i.e., "[0,pi,pi-,0+]".

    """
    # return        np.concatenate([X, np.conj(X[-2:0:-1])])
    # equivalent to np.concatenate([X, np.conj(np.flip(X[1:-1]))])
    flip_interior = np.flip(X[1:-1]) # negative-frequency part
    if is_magnitude:
        flip_conj_interior = flip_interior # negative-frequency part
    else:
        flip_conj_interior = np.conj(flip_interior)
    return np.concatenate([X, flip_conj_interior]) # complete spectrum
