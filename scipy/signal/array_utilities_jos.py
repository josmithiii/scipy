import numpy as np

def upsample_array(arr, factor):
    """
    Upsample a numpy array by an integer factor,
    keeping the first and last points unchanged.

    Parameters:
    arr (numpy.ndarray): The input array to upsample.
    factor (int): The upsampling factor. Must be an integer > 1.

    Returns:
    numpy.ndarray: The upsampled array.
    """
    if not isinstance(factor, int) or factor <= 1:
        raise ValueError("Upsampling factor must be an integer greater than 1")

    if arr.size < 2:
        return arr.copy()  # Nothing to upsample for arrays of size 0 or 1

    # Create the new array with the upsampled size
    new_size = (arr.size - 1) * factor + 1
    upsampled = np.zeros(new_size, dtype=arr.dtype)

    # Set the first and last points
    upsampled[0] = arr[0]
    upsampled[-1] = arr[-1]

    # Calculate the intermediate points
    for i in range(arr.size - 1):
        start_idx = i * factor  # Starting index for this interval
        end_idx = (i + 1) * factor  # Ending index for this interval
        # Generate `factor + 1` points between arr[i] and arr[i+1],
        # then take all but the first and last to fill the interval
        upsampled[start_idx:end_idx] = np.linspace(arr[i], arr[i+1], factor + 1)[0:-1]

    return upsampled
