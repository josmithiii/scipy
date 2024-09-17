import math
import numpy as np
from spectrum_utilities_jos import append_flip_conjugate
from scipy.fft import ifft, fft

from filter_plot_utilities_jos import (plot_mag_spectrum,
                                       plot_signal, plot_spectrum_overlay)

def check_roots_stability(roots, tol=1e-7):
    """
    Check the stability of roots.
    
    Args:
    roots (array-like): Array of complex roots
    tol (float): Tolerance for marginal stability
    
    Returns:
    tuple: (number of unstable roots, number of marginally stable roots)

    Example usage:
    roots = np.array([0.5 + 0.5j, 1.0001, 0.9999, 1.5, 0.8])
    num_unstable, num_marginally_stable = check_roots_stability(roots, tol=1e-3)
    print(f"Number of unstable roots: {num_unstable}")
    print(f"Number of marginally stable roots: {num_marginally_stable}")

    """
    magnitudes = np.abs(roots)
    num_unstable = np.sum(magnitudes > 1.0 + tol)
    num_marginally_stable = np.sum((magnitudes >= 1.0 - tol) &
                                   (magnitudes <= 1.0 + tol))
    return num_unstable, num_marginally_stable


def min_phase_spectrum(spec_lin_whole, n_fft, debug=False):
    n_fft_0 = len(spec_lin_whole) # whole spectrum, including negative frequencies
    if not math.log2(n_fft_0).is_integer():
        print(f"min_phase_spectrum: Warning: length of complete spectrum "
              f"{n_fft_0=} is not a power of 2")
    abs_spec_lin_whole = np.abs(spec_lin_whole)
    #E: log_spec = np.log(abs_spec_lin_whole
    #              + 1e-8 * np.max(abs_spec_lin_whole))
    #E: plot_mag_spec_lin_whole(log_spec,
    #              title="Log Magnitude Spectrum Needing Smoothing")
    spec_db_whole = 20 * np.log10(abs_spec_lin_whole
                                  + 1e-8 * np.max(abs_spec_lin_whole))
    if debug:
        plot_mag_spectrum(spec_db_whole,
                          title="DB Magnitude Spectrum Before Upsampling")
        # spec_db_whole_upsampled = resample(spec_db_whole, n_fft, domain='freq')
        print("*** USING SIMPLE LINEAR-INTERPOLATION FOR UPSAMPLING ***")
    # breakpoint()
    n_spec_0 = n_fft_0 // 2 + 1 # dc to fs/2 inclusive
    spec_db_half = spec_db_whole[ : n_spec_0 ]
    upsampling_factor = n_fft // n_fft_0
    spec_db_half_upsampled = upsample_array(spec_db_half,
                                            upsampling_factor ) # endpoints fixed
    spec_db_whole_upsampled = append_flip_conjugate(spec_db_half_upsampled)
    assert len(spec_db_whole_upsampled) == n_fft, "Spectral upsampling bug"
    if debug:
        plot_mag_spectrum(spec_db_whole_upsampled,
                          title="DB Magnitude Spectrum After Upsampling")
    c = ifft(spec_db_whole_upsampled).real # real cepstrum - real input detected?
    if debug:
        plot_signal(c, title="Real Cepstrum")
    # Check aliasing of cepstrum (in theory there is always some):
    cepstrum_aliasing_error_percent = 100 * np.linalg.norm(c[round(n_fft_0*0.9)
                                       :round(n_fft_0*1.1)]) / np.linalg.norm(c)
    if debug:
        print(f"Cepstral time-aliasing check: Outer 20% of cepstrum holds "
              f"{cepstrum_aliasing_error_percent:.2f} % of total rms")
    # Check if aliasing error is too high
    if cepstrum_aliasing_error_percent > 1.0:  # arbitrary limit
        plot_mag_spectrum(spec_db_whole_upsampled, title="Upsampled Log Spectrum")
        raise ValueError('Increase n_fft and/or smooth Sdb to shorten cepstrum')

    # Fold cepstrum to reflect non-min-phase zeros inside unit circle
    cf = np.zeros(n_fft, dtype=complex)
    cf[0] = c[0]
    n_spec = n_fft // 2 + 1 # non-negative freqs
    cf[1:n_spec-1] = c[1:n_spec-1] + c[n_fft-1:n_spec-1:-1]
    cf[n_spec-1] = c[n_spec-1]
    if debug:
        plot_signal(cf, title="Folded Real Cepstrum")

    # Compute minimum-phase spectrum
    Cf = fft(cf)
    # Cfrs = resample(Cf, n_fft_0, domain='freq') # use decimate instead?
    if debug:
        print("*** USING SIMPLE DECIMATION FOR DOWNSAMPLING ***")
    Cfrs = Cf[::upsampling_factor]
    #E: Smp = np.exp(Cfrs)  # minimum-phase spectrum
    spec_minphase_lin_whole = np.power(10, Cfrs/20)  # minimum-phase spectrum

    if debug:
        wT = np.linspace(0, np.pi, n_spec_0)
        spec_lin_half = spec_lin_whole[:n_spec_0]
        plot_spectrum_overlay(spec_lin_half, spec_minphase_lin_whole[:n_spec_0], wT,
                              "original and min-phase spectra", "original",
                              "min phase", log_freq=False)
        # plot_mag_spectrum(spec_db_whole,
        #                   title="DB Magnitude Spectrum Before Upsampling")

    return spec_minphase_lin_whole


def min_phase_half_spectrum(spec_lin_half, n_fft, debug=False):
    n_spec = len(spec_lin_half)
    if not math.log2(n_spec-1).is_integer():
        print(f"min_phase: Warning: length of non-negative-frequency spectrum "
              f"{n_spec=} is not a power of 2 plus 1")
    spec_lin_whole = append_flip_conjugate(np.abs(spec_lin_half), is_magnitude=True)
    assert n_fft > 2 * (n_spec-1), f"{n_fft=} should be larger than twice "
    f"spec_lin_half size + 1 = {2 * (n_spec-1)}"
    mps = min_phase_spectrum(spec_lin_whole, n_fft, debug=debug)
    Smpp = mps[:n_spec] # nonnegative-frequency portion
    return Smpp


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
