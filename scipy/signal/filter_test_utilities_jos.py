"""
Module/script name: filter_test_utilities_jos.py
Temporary test file for invfreqz proposal development.
Author: Julius Smith
Date: Started 9/03/24
Usage:
    
Dependencies:
    - scipy.signal
    - scipy.linalg
    - numpy
Additional notes:
    Intended not to be included in the final scipy squash-merge,
    but rather adapted into scipy unit tests which I've not yet learned about.
"""

import math
import numpy as np
from scipy.signal import freqz, resample
from scipy.linalg import norm
from scipy.fft import ifft, fft
#import matplotlib.pyplot as plt

# ./invfreqz_jos.py
from invfreqz_jos import (fast_equation_error_filter_design,
                          fast_steiglitz_mcbride_filter_design,
                          invert_unstable_roots, append_flip_conjugate)
from filter_plot_utilities_jos import plot_frequency_response_fit, plot_mag_spectrum


def maybe_stop():
    #breakpoint() # uncomment to stop
    pass


def min_phase_spectrum(spectrum, n_fft):
    n_fft_0 = len(spectrum) # whole spectrum, including negative frequencies
    if not math.log2(n_fft_0).is_integer():
        print(f"min_phase_spectrum: Warning: length of complete spectrum "
              f"{n_fft_0=} is not a power of 2")
    abs_spectrum = np.abs(spectrum)
    log_spec = np.log(abs_spectrum + 1e-8 * np.max(abs_spectrum))
    plot_mag_spectrum(log_spec, title="Log Magnitude Spectrum Needing Smoothing")
    breakpoint()
    log_spec_upsampled = resample(log_spec, n_fft, domain='freq')
    c = ifft(log_spec_upsampled).real # real cepstrum - real input detected?
    # Check aliasing of cepstrum (in theory there is always some):
    caliaserr = 100 * np.linalg.norm(c[round(n_fft_0*0.9)
                                       :round(n_fft_0*1.1)]) / np.linalg.norm(c)
    print(f"Cepstral time-aliasing check: Outer 20% of cepstrum holds "
          f"{caliaserr:.2f} % of total rms")

    # Check if aliasing error is too high
    if caliaserr > 1.0:  # arbitrary limit
        plot_mag_spectrum(log_spec_upsampled, title="Upsampled Log Spectrum")
        raise ValueError('Increase n_fft and/or smooth Sdb to shorten cepstrum')

    # Fold cepstrum to reflect non-min-phase zeros inside unit circle
    cf = np.zeros(n_fft, dtype=complex)
    cf[0] = c[0]
    n_spec = n_fft // 2 + 1 # non-negative freqs
    cf[1:n_spec-1] = c[1:n_spec-1] + c[n_fft-1:n_spec-1:-1]
    cf[n_spec-1] = c[n_spec-1]

    # Compute minimum-phase spectrum
    Cf = fft(cf)
    Cfrs = resample(Cf, n_fft_0, domain='freq') # use decimate instead?
    Smp = np.exp(Cfrs)  # minimum-phase spectrum

    return Smp


def min_phase_half_spectrum(half_spec, n_fft):
    n_spec = len(half_spec)
    if not math.log2(n_spec-1).is_integer():
        print(f"min_phase: Warning: length of non-negative-frequency spectrum "
              f"{n_spec=} is not a power of 2 plus 1")
    mag_spectrum = append_flip_conjugate(np.abs(half_spec), is_magnitude=True)
    assert n_fft > 2 * (n_spec-1), f"{n_fft=} should be larger than twice "
    f"half_spec size + 1 = {2 * (n_spec-1)}"
    mps = min_phase_spectrum(mag_spectrum, n_fft)
    Smpp = mps[:n_spec] # nonnegative-frequency portion
    return Smpp


def check_roots_stability(roots, tol=1e-7):
    magnitudes = np.abs(roots)
    num_unstable = np.sum(magnitudes > 1.0 + tol)
    num_marginally_stable = np.sum((magnitudes >= 1.0 - tol) &
                                   (magnitudes <= 1.0 + tol))
    return num_unstable, num_marginally_stable


def test_invfreqz(b, a, n_bh, n_ah, N, title, log_freq=False, try_iterative=False):
    print("--------------------------------------------------------------------------------")
    err = test_eqnerr(b, a, n_bh, n_ah, N, title, log_freq=log_freq)
    if try_iterative:
        maybe_stop()
        print("----------------------------")
        err += test_steiglitz_mcbride(b, a, n_bh, n_ah, N, title, log_freq=log_freq)
    return err

def test_eqnerr(b, a, n_bh, n_ah, N, title, log_freq=False):
    w = np.linspace(0, np.pi, int(N+1))
    w2,H = freqz(b, a, worN=w)
    if (not np.allclose(w,w2)):
        print("*** freqz is not computing frequencies as expected")

    if len(H) != N+1:
        print(f"*** freq response length is {len(H)} when we expected length {N+1}")
        exit

    U = np.ones_like(H)
    maybe_stop()
    bh, ah = fast_equation_error_filter_design(H, n_bh, n_ah, U=U )
    if not isinstance(bh, np.ndarray) or not isinstance(ah, np.ndarray):
        print("*** fast_equation_error_filter_design aborted")
        return

    print(f"{title}:")
    print("Original coefficients:")
    print(f"b = {b}")
    print(f"a = {a}")
    print("Estimated coefficients:")
    print(f"bh = {bh}")
    print(f"ah = {ah}")
    if len(b)-1 == n_bh and len(a)-1 == n_ah: # model-matched case
        print("Errors:")
        print(f"b-bh = {b-bh}")
        print(f"a-ah = {a-ah}")
        print("Total Coefficient Error:")
        error_coeffs = norm(a-ah) + norm(b-bh)
        print(f"norm(a-ah) + norm(b-ba) = {error_coeffs}")
    error_freq_resp = plot_frequency_response_fit(b, a, bh, ah, w, title,
                                                  log_freq=log_freq, show_plot=True)
    print(f"norm(frequency_response_error) = {error_freq_resp}")
    return error_freq_resp 


def test_steiglitz_mcbride(b, a, n_bh, n_ah, N, title, log_freq=False):
    print("Steiglitz McBride:")

    w = np.linspace(0, np.pi, int(N+1))
    w2,H = freqz(b, a, worN=w)
    U = np.ones_like(H)

    bh, ah = fast_steiglitz_mcbride_filter_design(
        H, U, n_bh, n_ah,
        max_iterations=30,
        tol_iteration_change=1e-12,
        initial_learning_rate=0.1)
    print(f"\n{title}:")
    print("Original coefficients:")
    print(f"b = {b}")
    print(f"a = {a}")
    print("Estimated coefficients:")
    print(f"bh = {bh}")
    print(f"ah = {ah}")
    if len(b)-1 == n_bh and len(a)-1 == n_ah:
        print("Errors:")
        print(f"b-bh = {b-bh}")
        print(f"a-ah = {a-ah}")
        print("Total Coefficient Error:")
        error_coeffs = norm(a-ah) + norm(b-bh)
        print(f"norm(a-ah) + norm(b-ba) = {error_coeffs}")
    print("--------------------------------------------------------------")
    print("Stabilize if needed:")
    ah_stable, roots, ah_stable = invert_unstable_roots(ah)
    print("Filter poles:", roots)
    print("Filter pole magnitudes:", np.abs(roots))
    if not ah_stable:
        print("Filter design UNSTABLE!")
        print("Original polynomial coefficients:", a)
        print("Stabilized polynomial coefficients:", ah_stable)
        print("Roots after inversion:", roots)
    else:
        print("Filter design is not unstable")

    num_unstable, num_marginally_stable = check_roots_stability(roots,tol=1e-7)

    if num_marginally_stable > 0:
        print(f"""
        {num_marginally_stable} MARGINALLY UNSTABLE poles
        (within 1e-7 of radius 1.0)
        """)

    if num_unstable > 0:
        print(f"""
        *** {num_unstable} UNSTABLE POLES found
        _after_ calling invert_unstable_roots
        """)

    error_freq_resp = plot_frequency_response_fit(b, a, bh, ah, w, title,
                                                  show_plot=True, log_freq=log_freq)
    print(f"norm(frequency_response_error) = {error_freq_resp}")

    return error_freq_resp 
