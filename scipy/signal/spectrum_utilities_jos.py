import math
import numpy as np
from scipy.fft import ifft, fft

from array_utilities_jos import upsample_array
from spectrum_plot_utilities_jos import (plot_mag_spectrum,
                                         plot_signal, plot_spectrum_overlay)

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
