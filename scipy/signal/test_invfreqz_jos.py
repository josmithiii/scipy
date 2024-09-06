import numpy as np
# from scipy import signal
from scipy.signal import freqz, butter, cheby1
from scipy.linalg import norm
import matplotlib.pyplot as plt
from invfreqz_jos import (fast_equation_error_filter_design,
                          fast_steiglitz_mcbride_filter_design,
                          invert_unstable_roots)  # ./invfreqz_jos.py
from filter_plot_utilities_jos import plot_frequency_response_fit # , zplane

import pprint
pp = pprint.PrettyPrinter(indent=4)

# pytest --cache-clear

# import pdb
# pdb.set_trace()

def maybe_stop():
    #breakpoint() # uncomment to stop
    pass

def run_test(b, a, n_bh, n_ah, N, title):
    #orig: w = np.logspace(-3, np.log10(np.pi), N)
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
    plot_frequency_response_fit(b, a, bh, ah, w, title)
    maybe_stop()
    print("--------------------------------------------------------------")
    print("Steiglitz McBride:")

    breakpoint()

    bh, ah = fast_steiglitz_mcbride_filter_design(
        H, U, n_bh, n_ah,
        max_iterations=30, tol_iteration_change=1e-12)
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
        print("Total Error:")
        print(f"norm(a-ah) + norm(b-ba) = {norm(a-ah) + norm(b-bh)}")
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

    def check_roots_stability(roots, tol=1e-7):
        magnitudes = np.abs(roots)
        num_unstable = np.sum(magnitudes > 1.0 + tol)
        num_marginally_stable = np.sum((magnitudes >= 1.0 - tol) &
                                       (magnitudes <= 1.0 + tol))
        return num_unstable, num_marginally_stable

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

def get_orders(n_b, n_a, model_complete_case=True):
    if model_complete_case:
        n_bh = n_b
        n_ah = n_a
    else:
        n_bh = n_b - 1 # one too few zeros
        n_ah = n_a - 1 # one too few poles
    return n_bh, n_ah

################################# TESTS ##############################
print("--------------------------------------------------------------")

model_complete_case = True  # In the model-complete case,
                            # the filter design can exactly match the desired
# model_complete_case = False # force an approximate design

# # Test 0: Butterworth lowpass filter order 1
# N = 8
# b_butter, a_butter = butter(1, 0.25, btype='low', analog=False)
# run_test(b_butter, a_butter, 1, 1, N, "Butterworth Lowpass Filter")
# plt.show()
# #exit();

print("------------------------------------------------------------------------------")
print("Test 0.1: Butterworth lowpass filter order 2")
N = 16
n_b = 2
n_a = 2
b_butter, a_butter = butter(n_a, 0.25, btype='low', analog=False)
# n_bh, n_ah = get_orders(n_b, n_a, model_complete_case) # smaller if not model-complete
print("-------------------------- Model Complete --------------------------------")
run_test(b_butter, a_butter, n_b, n_a, N, "Butterworth Lowpass Filter")
plt.show()
print("------------------------- Model Incomplete -------------------------------")
run_test(b_butter, a_butter, n_b-1, n_a-1, N, "Reduced-Order Butterworth Lowpass")
plt.show()
exit()

# # Test 0.2: Butterworth lowpass filter order 3
# N = 16
# b_butter, a_butter = butter(3, 0.25, btype='low', analog=False)
# run_test(b_butter, a_butter, 3, 3, N, "Butterworth Lowpass Filter")
# plt.show()
# exit();

# # Test 0.3: Pathological unstable max-phase target:
b_path = [1, 2, 3, 2, 3]
a_path = [1, 2, 3, 2, 1, 4]
N = 64
run_test(b_path, a_path, 4, 5, N, "Pathological Target Filter")
plt.show()
#exit();

# Test 1: Butterworth lowpass filter
N = 1024
b_butter, a_butter = butter(4, 0.2, btype='low', analog=False)
run_test(b_butter, a_butter, 4, 4, N, "Butterworth Lowpass Filter")
plt.show()
#exit();

# Test 2: Chebyshev Type I bandpass filter
b_cheby, a_cheby = cheby1(6, 1, [0.25, 0.75], btype='band', analog=False)
run_test(b_cheby, a_cheby, 12, 12, N, "Chebyshev Type I Bandpass Filter")

# Test 3: Custom filter with multiple poles and zeros
b_custom = [0.0255, 0.0510, 0.0255]
a_custom = [1.0, -1.3790, 0.5630]
run_test(b_custom, a_custom, 2, 2, N, "Custom Filter")

plt.show()
