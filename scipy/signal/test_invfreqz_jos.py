"""
Module/script name: test_invfreqz_jos.py
Temporary test file for invfreqz proposal development.
Author: Julius Smith
Date: Started 9/03/24
Usage:
    
Dependencies:
    - sys
    - os
    - scipy.signal
    - filter_test_utilities_jos
Additional notes:
    Intended not to be included in the final scipy squash-merge,
    but rather adapted into scipy unit tests which I've not yet learned about.
"""

import sys
import os
import numpy as np

# from scipy import signal
from scipy.signal import butter, cheby1, minimum_phase

from filter_test_utilities_jos import test_invfreqz
            # , test_eqnerr, test_steiglitz_mcbride, test_prony, test_pade_prony

# pytest --cache-clear

# import pdb
# pdb.set_trace()

if len(sys.argv) == 2:
    try:
        test_num = int(sys.argv[1])
        print(f"Test {test_num}:")
    except ValueError:
        print("Please provide a valid integer n = test number, or 0 to run all tests")
        exit()
else:
    test_num = 0
        
total_error = 0

if test_num == 1 or test_num == 0:
    order = 1
    n_freqs = 8
    b_butter_1, a_butter_1 = butter(order, 0.25, btype='low', analog=False)
    label = f"{test_num}: Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_1, a_butter_1, order, order, n_freqs, label)

if test_num == 2 or test_num == 0:
    order = 2
    n_freqs = 16
    b_butter_2, a_butter_2 = butter(order, 0.25, btype='low', analog=False)
    label_complete = f"Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    test_invfreqz(b_butter_2, a_butter_2, order, order, n_freqs, label_complete)
    order_reduced = order - 1
    label_reduced = "Reduced-Order Butterworth lowpass, "
    f"order {order_reduced}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_2, a_butter_2, order_reduced, order_reduced,
                                 n_freqs, label_reduced)

if test_num == 3 or test_num == 0:
    order = 3
    n_freqs = 16
    b_butter_3, a_butter_3 = butter(order, 0.25, btype='low', analog=False)
    label = f"{test_num}: Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_3, a_butter_3, order, order, n_freqs, label)

if test_num == 4 or test_num == 0:
    order = 4
    n_freqs = 1024
    b_butter_4, a_butter_4 = butter(order, 0.2, btype='low', analog=False)
    label = f"{test_num}: Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_4, a_butter_4, order, order, n_freqs, label)

if test_num == 5 or test_num == 0:
    b_path = [1, 2, 3, 2, 3]
    n_b = len(b_path)-1
    a_path = [1, 2, 3, 2, 1, 4]
    n_a = len(a_path)-1 # order
    order = max(n_b, n_a)
    n_freqs = 64
    label = f"{test_num}: Pathological unstable max-phase target,"
    f" order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_path, a_path, n_b, n_a, n_freqs, label)

if test_num == 6 or test_num == 0:
    N = 1024
    proto_order = 6
    order = 2 * proto_order
    label = f"{test_num}: Chebyshev Type I bandpass filter, order {order}, n_freq {N}"
    b_cheby, a_cheby = cheby1(proto_order, 1, [0.25, 0.75], btype='band', analog=False)
    total_error += test_invfreqz(b_cheby, a_cheby, order, order, N, label)

if test_num == 7 or test_num == 0:
    n_freq = 1024
    label = "Custom filter with multiple poles and zeros"
    b_custom = [0.0255, 0.0510, 0.0255]
    a_custom = [1.0, -1.3790, 0.5630]
    total_error += test_invfreqz(b_custom, a_custom, 2, 2, n_freq, label)

if test_num == 8 or test_num == 0:
    n_freq = 1024
    print("1/f filter")
    indices = np.arange(n_freq+1)
    power = 0.5 # 1/sqrt(f)
    rolloff = 1 / np.power(indices + 1, power) # 1/(n+1)^p
    n_fft = 4 * n_freq
    rolloff_mp = minimum_phase(rolloff, method='homomorphic',
                               n_fft=n_fft) #, half=False)
    b_rolloff = np.fft.ifft(rolloff_mp)
    a_rolloff = np.ones(1)
    n_b = len(b_rolloff)-1
    n_a = len(a_rolloff)-1
    # order = max(n_b, n_a) # model-complete
    order = 4 # reduced-order approximation
    label = f"{test_num}: 1/f^{power} rolloff filter, order {order}, n_freq {n_freq}"
    total_error += test_invfreqz(b_rolloff, a_rolloff, order, order,
                                 n_freq, label, log_freq=True)

# -------------------------------------------------------------------------------

if test_num == 0:
    print("--------------------------------------------------------------------------------")
    this_file = os.path.basename(__file__)
    print(f"{this_file}:\n Sum of all test errors is {total_error}")
