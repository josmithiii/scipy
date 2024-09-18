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
    Intended not to be included in the final scipy pull-request squash-merge,
    but rather adapted into scipy unit tests which I've not yet learned about.
"""

import numpy as np
# from scipy.signal import freqz # , resample
from scipy.linalg import norm
#import matplotlib.pyplot as plt

# ./invfreqz_jos.py
from invfreqz_jos import (fast_equation_error_filter_design,
                          fast_steiglitz_mcbride_filter_design,
                          invert_unstable_roots)
from filter_plot_utilities_jos import (plot_frequency_response_fit, get_freq_response)
from filter_utilities_jos import (check_roots_stability)

def maybe_stop():
    #breakpoint() # uncomment to stop
    pass


def test_invfreqz(b, a, n_bh, n_ah, n_spec, title, log_freq=False,
                  n_iter=0, debug=False):
    print("--------------------------------------------------------------------------------")
    if n_iter == 0:
        err = test_eqnerr(b, a, n_bh, n_ah, n_spec, title,
                          log_freq=log_freq, debug=debug)
    else:
        err = test_steiglitz_mcbride(b, a, n_bh, n_ah, n_spec, title, n_iter=n_iter,
                                     log_freq=log_freq, debug=debug)
    return err


def test_eqnerr(b, a, n_bh, n_ah, n_spec, title, log_freq=False, debug=False):

    w, H, have_truth = get_freq_response(b, a, n_spec)
    U = np.ones_like(H)
    maybe_stop()
    bh, ah = fast_equation_error_filter_design(H, n_bh, n_ah, U=U )
    if not isinstance(bh, np.ndarray) or not isinstance(ah, np.ndarray):
        print("*** fast_equation_error_filter_design aborted")
        return

    print(f"{title}:")
    if have_truth:
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


def test_steiglitz_mcbride(b, a, n_bh, n_ah, n_spec, title, n_iter=5,
                           log_freq=False, debug=False):
    print("Steiglitz McBride:")

    w, H, have_truth = get_freq_response(b, a, n_spec)
    U = np.ones_like(H)

    bh, ah = fast_steiglitz_mcbride_filter_design(
        H, U, n_bh, n_ah,
        n_iter=n_iter,
        tol_iter=1e-12,
        lr0=0.1)
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
