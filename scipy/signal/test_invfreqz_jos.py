
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
    - filter_plot_utilities_jos
Additional notes:
    Intended not to be included in the final scipy pull-request squash-merge,
    but rather adapted into scipy unit tests which I've not yet learned about.
"""

import sys
import os
import numpy as np

# from scipy import signal
from scipy.signal import butter, cheby1, freqz

from filter_plot_utilities_jos import dB, plot_mag_spectrum
from filter_test_utilities_jos import test_invfreqz, min_phase_half_spectrum
            # , test_eqnerr, test_steiglitz_mcbride, test_prony, test_pade_prony

from invfreqz_jos import append_flip_conjugate

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
    total_error += test_invfreqz(b_butter_2, a_butter_2, order, order, n_freqs,
                                 label_complete)
    order_reduced = order - 1
    label_reduced = "Reduced-Order Butterworth lowpass, "
    f"order {order_reduced}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_2, a_butter_2, order_reduced, order_reduced,
                                 n_freqs, label_reduced)
    total_error += test_invfreqz(b_butter_2, a_butter_2, order_reduced, order_reduced,
                                 n_freqs, label_reduced, n_iter=5)


if test_num == 3 or test_num == 0:
    order = 2
    n_freqs = 16
    b_res_2 = np.array([1, 0, -1], dtype=float)
    R = 0.9
    a_res_2 = np.array([1, 0, R*R], dtype=float)
    label_res = f"pi/2 resonator, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_res_2, a_res_2, order, order, n_freqs, label_res)
    total_error += test_invfreqz(b_res_2, a_res_2, order, order,
                                 n_freqs, label_res, n_iter=5)

    nl = 0.1 # Add "white noise" floor
    b_respn_2 = b_res_2 + nl * a_res_2
    a_respn_2 = a_res_2
    label_pn = f"pi/2 resonator plus {nl}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_respn_2, a_respn_2, order, order,
                                 n_freqs, label_pn)
    total_error += test_invfreqz(b_respn_2, a_respn_2, order, order,
                                 n_freqs, label_pn, n_iter=5)


if test_num == 4 or test_num == 0:
    order = 3
    n_freqs = 16
    b_butter_3, a_butter_3 = butter(order, 0.25, btype='low', analog=False)
    label = f"{test_num}: Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_3, a_butter_3, order, order, n_freqs, label)
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
    total_error += test_invfreqz(b_path, a_path, n_b, n_a, n_freqs, label,
                                 n_iter=30, debug=False)

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

def self_convolve(arr, p):
    if not isinstance(p, (int | float)):
        raise TypeError("p must be a number")

    if not p.is_integer() or p < 1:
        raise ValueError("p must be a positive integer")

    p = int(p)  # Convert p to int for use in range()

    result = arr.copy()
    for _ in range(p - 1):
        result = np.convolve(result, arr, mode='full')
    return result

def model_matched_rolloff(power, test_num, title):
    print("--------------------------------------------------------------------------------")
    assert power.is_integer() and power > 0, f"{power=} must be a positive integer"
    n_freq = 1024
    wT = np.linspace(0, np.pi, n_freq+1)
    pole = 0.999
    rolloff_denom = self_convolve([1, -pole], power)
    print(f"{rolloff_denom=}")
    _,rolloff_mp = freqz( (1-pole) ** power, rolloff_denom, worN=wT)
    plot_mag_spectrum(dB(rolloff_mp),
                      title=f"{int(power)}-pole magnitude frequency response",
                      mag_units='dB')
    b_rolloff = np.fft.ifft(append_flip_conjugate(rolloff_mp))
    a_rolloff = np.ones(1)
    order = int(power) # exact match possible
    n_b = order
    n_a = order
    label = f"{test_num}: {title} filter, order {order}, n_freq {n_freq}"
    error = test_invfreqz(b_rolloff, a_rolloff, n_b, n_a,
                                 n_freq, label, log_freq=True)
    return error

if test_num == 8 or test_num == 0:
    title = "1/f rolloff"
    print(title)
    power = 1.0 # model-matched case
    total_error += model_matched_rolloff(power, test_num, title)

if test_num == 9 or test_num == 0:
    title = "1/f^2 rolloff"
    power = 2.0 # model-matched case
    total_error += model_matched_rolloff(power, test_num, title)

if test_num == 10 or test_num == 0:
    title = "1/f^3 rolloff"
    power = 3.0 # model-matched case
    total_error += model_matched_rolloff(power, test_num, title)

def model_incomplete_rolloff(power, test_num, title=None, n_freq=1024):
    print("--------------------------------------------------------------------------------")
    if title is None:
        title = f"1/f^{power} rolloff"

    # Create desired magnitude spectrum, from dc to fs/2 inclusive:
    indices = 1 + np.arange( n_freq )
    rolloff_lin_half = np.power( indices, -power )
    rolloff_lin_half = np.concatenate(([rolloff_lin_half[0]],
                                       rolloff_lin_half)) # [1, 1/(n+1)^p], n=0,1,...

    # plot it:
    wT = np.linspace(0, np.pi, n_freq+1)
    rolloff_db_half = dB(rolloff_lin_half)
    plot_mag_spectrum(rolloff_db_half, wT=wT,
                      title=f"{title} magnitude frequency response, "
                      "pre-interpolation",
                      mag_units='dB')
    n_fft = 4 * n_freq # for spectral interpolation
    rolloff_mp_lin_half = min_phase_half_spectrum(rolloff_lin_half,
                                                  n_fft=n_fft) #, half=False)
    rolloff_mp_lin_whole = append_flip_conjugate(rolloff_mp_lin_half)
    b_rolloff = np.fft.ifft(rolloff_mp_lin_whole)
    a_rolloff = np.ones(1)
    n_b = 4
    n_a = 4
    label = f"{test_num}: 1/f^{power} rolloff filter, {n_a=} {n_b=}, n_freq {n_freq}"
    error1 = test_invfreqz(b_rolloff, a_rolloff, n_b, n_a,
                          n_freq, label, log_freq=True)
    error2 = test_invfreqz(b_rolloff, a_rolloff, n_b, n_a,
                           n_freq, label, log_freq=True, n_iter=5)
    return error1 + error2

if test_num == 11 or test_num == 0:
    power = 0.5
    total_error += model_incomplete_rolloff(power, test_num)

if test_num == 12 or test_num == 0:
    power = 1.5
    total_error += model_incomplete_rolloff(power, test_num)

# -------------------------------------------------------------------------------

if test_num == 0:
    print("--------------------------------------------------------------------------------")
    this_file = os.path.basename(__file__)
    print(f"{this_file}:\n Sum of all test errors is {total_error}")
