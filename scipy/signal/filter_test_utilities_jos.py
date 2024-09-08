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

from scipy.signal import freqz
from scipy.linalg import norm
import numpy as np

from invfreqz_jos import (fast_equation_error_filter_design,
                          fast_steiglitz_mcbride_filter_design,
                          invert_unstable_roots)  # ./invfreqz_jos.py
from filter_plot_utilities_jos import plot_frequency_response_fit # , zplane

def maybe_stop():
    #breakpoint() # uncomment to stop
    pass


def check_roots_stability(roots, tol=1e-7):
    magnitudes = np.abs(roots)
    num_unstable = np.sum(magnitudes > 1.0 + tol)
    num_marginally_stable = np.sum((magnitudes >= 1.0 - tol) &
                                   (magnitudes <= 1.0 + tol))
    return num_unstable, num_marginally_stable


def test_invfreqz(b, a, n_bh, n_ah, N, title, try_iterative=False):
    print("--------------------------------------------------------------------------------")
    err = test_eqnerr(b, a, n_bh, n_ah, N, title)
    if try_iterative:
        maybe_stop()
        print("----------------------------")
        err += test_steiglitz_mcbride(b, a, n_bh, n_ah, N, title)
    return err

def test_eqnerr(b, a, n_bh, n_ah, N, title):
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
    error_freq_resp = plot_frequency_response_fit(b, a, bh, ah, w, title)
    print(f"norm(frequency_response_error) = {error_freq_resp}")
    return error_freq_resp 


def test_steiglitz_mcbride(b, a, n_bh, n_ah, N, title):
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

    error_freq_resp = plot_frequency_response_fit(b, a, bh, ah, w, title)
    print(f"norm(frequency_response_error) = {error_freq_resp}")

    return error_freq_resp 
