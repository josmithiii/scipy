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

from spectrum_plot_utilities_jos import dB, plot_mag_spectrum
from spectrum_utilities_jos import min_phase_half_spectrum # , append_flip_conjugate
from filter_test_utilities_jos import test_invfreqz

# pytest --cache-clear
# Testing: cd /w/scipy && pytest --cache-clear

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
    total_error += test_invfreqz(b_butter_1, a_butter_1, order, order, n_freqs+1, label)

if test_num == 2 or test_num == 0:
    order = 2
    n_freqs = 16
    b_butter_2, a_butter_2 = butter(order, 0.25, btype='low', analog=False)
    label_complete = f"Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_2, a_butter_2, order, order, n_freqs+1,
                                 label_complete)
    order_reduced = order - 1
    label_reduced = "Reduced-Order Butterworth lowpass, "
    f"order {order_reduced}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_2, a_butter_2, order_reduced, order_reduced,
                                 n_freqs+1, label_reduced)
    total_error += test_invfreqz(b_butter_2, a_butter_2, order_reduced, order_reduced,
                                 n_freqs+1, label_reduced, n_iter=5)


if test_num == 3 or test_num == 0:
    order = 2
    n_freqs = 16
    b_res_2 = np.array([1, 0, -1], dtype=float)
    R = 0.9
    a_res_2 = np.array([1, 0, R*R], dtype=float)
    label_res = f"pi/2 resonator, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_res_2, a_res_2, order, order, n_freqs+1, label_res)
    total_error += test_invfreqz(b_res_2, a_res_2, order, order,
                                 n_freqs+1, label_res, n_iter=5)

    nl = 0.1 # Add "white noise" floor
    b_respn_2 = b_res_2 + nl * a_res_2
    a_respn_2 = a_res_2
    label_pn = f"pi/2 resonator plus {nl}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_respn_2, a_respn_2, order, order,
                                 n_freqs+1, label_pn)
    total_error += test_invfreqz(b_respn_2, a_respn_2, order, order,
                                 n_freqs+1, label_pn, n_iter=5)


if test_num == 4 or test_num == 0:
    order = 3
    n_freqs = 16
    b_butter_3, a_butter_3 = butter(order, 0.25, btype='low', analog=False)
    label = f"{test_num}: Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_3, a_butter_3, order, order, n_freqs+1, label)
    order = 4
    n_freqs = 1024
    b_butter_4, a_butter_4 = butter(order, 0.2, btype='low', analog=False)
    label = f"{test_num}: Butterworth lowpass filter, order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_butter_4, a_butter_4, order, order, n_freqs+1, label)

if test_num == 5 or test_num == 0:
    b_path = [1, 2, 3, 2, 3]
    n_b = len(b_path)-1
    a_path = [1, 2, 3, 2, 1, 4]
    n_a = len(a_path)-1 # order
    order = max(n_b, n_a)
    n_freqs = 64
    label = f"{test_num}: Pathological unstable max-phase target,"
    f" order {order}, n_freqs {n_freqs}"
    total_error += test_invfreqz(b_path, a_path, n_b, n_a, n_freqs+1, label)
    total_error += test_invfreqz(b_path, a_path, n_b, n_a, n_freqs+1, label,
                                 n_iter=15, debug=False)

if test_num == 6 or test_num == 0:
    n_freqs = 1024
    proto_order = 6
    order = 2 * proto_order
    label = f"{test_num}: Chebyshev Type I bandpass filter, order {order},"
    f" n_freqs {n_freqs}"
    b_cheby, a_cheby = cheby1(proto_order, 1, [0.25, 0.75], btype='band', analog=False)
    total_error += test_invfreqz(b_cheby, a_cheby, order, order, n_freqs+1, label)

if test_num == 7 or test_num == 0:
    n_freqs = 1024
    label = "Custom filter with multiple poles and zeros"
    b_custom = [0.0255, 0.0510, 0.0255]
    a_custom = [1.0, -1.3790, 0.5630]
    total_error += test_invfreqz(b_custom, a_custom, 2, 2, n_freqs+1, label)

# ---------------- Hand Crafted Low/High/Band Pass/Stop Filters ------------------------

n_spec_hc = 513
M_hc = 5   # Numerator order
N_hc = 5   # Denominator order

if test_num == 20 or test_num == 0:
    label = "Ideal Lowpass Filter"
    cutoff_lp = np.pi / 2
    H_lp = np.zeros(n_spec_hc)
    freqs = np.linspace(0, np.pi, n_spec_hc) # 0 and pi included
    H_lp[freqs < cutoff_lp] = 1
    total_error += test_invfreqz(H_lp, 1, M_hc, N_hc, n_spec_hc, label)

    # Min Phase Ideal Low-pass Filter
    label = "Test 20B: Min Phase Ideal Low-pass Filter"
    n_fft_hc = 4 * (n_spec_hc-1)
    H_lp_mp = min_phase_half_spectrum(H_lp, n_fft_hc, debug=False)
    total_error += test_invfreqz(H_lp_mp, 1, M_hc, N_hc, n_spec_hc, label)

    # Min Phase Ideal-ROLLOFF Low-pass Filter: 1/freq^order rolloff
    label = "Test 20C: Min Phase Ideal ROLLOFF Low-pass Filter"
    H_lp_ro = np.ones(n_spec_hc)
    rolloff_indices = freqs > cutoff_lp
    H_lp_ro[rolloff_indices] = (cutoff_lp / freqs[rolloff_indices]) ** N_hc
    H_lp_mp_ro = min_phase_half_spectrum(H_lp_ro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_lp_mp_ro, 1, M_hc, N_hc, n_spec_hc, label)

    label = "Test 20D: Min Phase Ideal FASTER Rolloff Low-pass Filter"
    H_lp_fro = np.ones(n_spec_hc)
    rolloff_indices = freqs > cutoff_lp
    H_lp_fro[rolloff_indices] = (cutoff_lp / freqs[rolloff_indices]) ** (2*N_hc)
    H_lp_mp_fro = min_phase_half_spectrum(H_lp_fro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_lp_mp_fro, 1, M_hc, N_hc, n_spec_hc, label)
    

if test_num == 21 or test_num == 0:
    label = "Ideal Highpass Filter"
    cutoff_hp = np.pi / 2
    H_hp = np.ones(n_spec_hc)
    freqs = np.linspace(0, np.pi, n_spec_hc) # 0 and pi included
    H_hp[freqs < cutoff_hp] = 0
    total_error += test_invfreqz(H_hp, 1, M_hc, N_hc, n_spec_hc, label)

    H_hp_mp = min_phase_half_spectrum(H_hp, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_hp_mp, 1, M_hc, N_hc, n_spec_hc, label)


    # Test Case 2: Ideal High-pass Filter
    label = "Test 2: Ideal High-pass Filter"
    cutoff_hp = np.pi / 2
    H_hp = np.zeros(n_spec_hc)
    H_hp[freqs > cutoff_hp] = 1

    if 0:
        n_spec_hc = 17 # ******** TEMPORARY HACK **********
        freqs = np.linspace(0, np.pi, n_spec_hc)

    label = "Test 2B: Min Phase Ideal Rolloff High-pass Filter"
    H_hp_ro = np.ones(n_spec_hc)
    rolloff_indices = freqs < cutoff_hp
    H_hp_ro[rolloff_indices] = (freqs[rolloff_indices] / cutoff_hp) ** N_hc
    H_hp_mp_ro = min_phase_half_spectrum(H_hp_ro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_hp_mp_ro, 1, M_hc, N_hc, n_spec_hc, label)

    label = "Test 2C: Min Phase Ideal Faster Rolloff High-pass Filter"
    H_hp_fro = np.ones(n_spec_hc)
    rolloff_indices = freqs < cutoff_hp
    H_hp_fro[rolloff_indices] = (freqs[rolloff_indices] / cutoff_hp) ** (2 * N_hc)
    H_hp_mp_fro = min_phase_half_spectrum(H_hp_fro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_hp_mp_fro, 1, M_hc, N_hc, n_spec_hc, label)


if test_num == 22 or test_num == 0:
    label = "Test 22: Ideal Band-pass Filter"
    band_start = np.pi / 4
    band_end = 3 * np.pi / 4
    H_bp = np.zeros(n_spec_hc)
    freqs = np.linspace(0, np.pi, n_spec_hc) # 0 and pi included
    H_bp[(freqs >= band_start) & (freqs <= band_end)] = 1
    total_error += test_invfreqz(H_bp, 1, M_hc, N_hc, n_spec_hc, label)

    label = "Test 22B: Min Phase Ideal Rolloff Band-Pass Filter"
    H_bp_ro = np.ones(n_spec_hc)
    rolloff_indices_lp = freqs > band_end
    rolloff_indices_hp = freqs < band_start
    H_bp_ro[rolloff_indices_lp] = ( band_end
                                    / freqs[rolloff_indices_lp] ) ** (N_hc//2)
    H_bp_ro[rolloff_indices_hp] = ( freqs[rolloff_indices_hp]
                                    / band_start ) ** (N_hc//2)
    H_bp_mp_ro = min_phase_half_spectrum(H_bp_ro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_bp_mp_ro, 1, M_hc, N_hc, n_spec_hc, label)

    label = "Test 22B: Min Phase Ideal Faster Rolloff Band-Pass Filter"
    H_bp_fro = np.ones(n_spec_hc)
    rolloff_indices_lp = freqs > band_end
    rolloff_indices_hp = freqs < band_start
    H_bp_fro[rolloff_indices_lp] = ( band_end / freqs[rolloff_indices_lp] ) ** (N_hc)
    H_bp_fro[rolloff_indices_hp] = ( freqs[rolloff_indices_hp] / band_start ) ** (N_hc)
    H_bp_mp_fro = min_phase_half_spectrum(H_bp_fro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_bp_mp_fro, 1, M_hc, N_hc, n_spec_hc, label)

if test_num == 23 or test_num == 0:
    label = "Test 23: Ideal Band-stop Filter"
    band_start = np.pi / 4
    band_end = 3 * np.pi / 4
    H_bs = np.ones(n_spec_hc)
    freqs = np.linspace(0, np.pi, n_spec_hc) # 0 and pi included
    H_bs[(freqs >= band_start) & (freqs <= band_end)] = 0

    label = "Test 23B: Min Phase Ideal Rolloff Band-Stop Filter"
    H_bs_ro = np.ones(n_spec_hc)
    rolloff_indices = (freqs >= band_start) & (freqs <= band_end)
    true_indices = np.where(rolloff_indices)[0] # All True indices
    middle_index = len(true_indices) // 2 # middle index
    icm = true_indices[middle_index] # index closest to middle
    print(f"index_closest_to_middle = {icm}")
    rolloff_indices_lp = (freqs >= band_start) & (freqs <= freqs[icm])
    rolloff_indices_hp = (freqs <= band_end  ) & (freqs >  freqs[icm])
    H_bs_ro[rolloff_indices_lp] = (band_start / freqs[rolloff_indices_lp]) ** (N_hc//2)
    H_bs_ro[rolloff_indices_hp] = (freqs[rolloff_indices_hp] / band_end) ** (N_hc//2)
    H_bs_mp_ro = min_phase_half_spectrum(H_bs_ro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_bs_mp_ro, 1, M_hc, N_hc, n_spec_hc, label)

    label = "Test 23C: Min Phase Ideal Faster Rolloff Band-Stop Filter"
    H_bs_fro = np.ones(n_spec_hc)
    rolloff_indices = (freqs >= band_start) & (freqs <= band_end)
    true_indices = np.where(rolloff_indices)[0] # All True indices
    middle_index = len(true_indices) // 2 # middle index
    icm = true_indices[middle_index] # index closest to middle
    print(f"index_closest_to_middle = {icm}")
    rolloff_indices_lp = (freqs >= band_start) & (freqs <= freqs[icm])
    rolloff_indices_hp = (freqs <= band_end  ) & (freqs >  freqs[icm])
    H_bs_fro[rolloff_indices_lp] = (band_start / freqs[rolloff_indices_lp]) ** (N_hc)
    H_bs_fro[rolloff_indices_hp] = (freqs[rolloff_indices_hp] / band_end) ** (N_hc)
    H_bs_mp_fro = min_phase_half_spectrum(H_bs_fro, 4 * (n_spec_hc-1), debug=False)
    total_error += test_invfreqz(H_bs_mp_fro, 1, M_hc, N_hc, n_spec_hc, label)


# ---------------------------- Rolloff Filter Designs (RFD) ---------------------------
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
    print("---------------------------------------------------------------------------")
    assert power.is_integer() and power > 0, f"{power=} must be a positive integer"
    n_spec = 1025 # fft_size // 2 + 1
    wT = np.linspace(0, np.pi, n_spec)
    pole = 0.999
    rolloff_denom = self_convolve([1, -pole], power)
    print(f"{rolloff_denom=}")
    _,rolloff_mp = freqz( (1-pole) ** power, rolloff_denom, worN=wT)
    plot_mag_spectrum(dB(rolloff_mp),
                      title=f"{int(power)}-pole magnitude frequency response",
                      mag_units='dB')
    # previous hack:
    # rolloff_mp_lin_whole = append_flip_conjugate(rolloff_mp_lin_half)
    # b_rolloff = np.fft.ifft(rolloff_mp_lin_whole)
    # new hack:
    b_rolloff = rolloff_mp
    a_rolloff = 1 # np.ones(1)
    order = int(power) # exact match possible
    n_b = order
    n_a = order
    label = f"{test_num}: {title} filter, order {order}, n_spec {n_spec}"
    error = test_invfreqz(b_rolloff, a_rolloff, n_b, n_a,
                                 n_spec, label, log_freq=True)
    return error

if test_num == 30 or test_num == 0:
    title = "1/f rolloff"
    print(title)
    power = 1.0 # model-matched case
    total_error += model_matched_rolloff(power, test_num, title)

if test_num == 31 or test_num == 0:
    title = "1/f^2 rolloff"
    power = 2.0 # model-matched case
    total_error += model_matched_rolloff(power, test_num, title)

if test_num == 32 or test_num == 0:
    title = "1/f^3 rolloff"
    power = 3.0 # model-matched case
    total_error += model_matched_rolloff(power, test_num, title)

def model_incomplete_rolloff(power, test_num, title=None, n_spec=1025):
    print("--------------------------------------------------------------------------------")
    if title is None:
        title = f"1/f^{power} rolloff"

    # Create desired magnitude spectrum, from dc to fs/2 inclusive:
    indices = 1 + np.arange( n_spec-1 )
    rolloff_lin_half = np.power( indices, -power )
    rolloff_lin_half = np.concatenate(([rolloff_lin_half[0]],
                                       rolloff_lin_half)) # [1, 1/(n+1)^p], n=0,1,...

    # plot it:
    wT = np.linspace(0, np.pi, n_spec)
    rolloff_db_half = dB(rolloff_lin_half)
    plot_mag_spectrum(rolloff_db_half, wT=wT,
                      title=f"{title} magnitude frequency response, "
                      "pre-interpolation",
                      mag_units='dB')
    n_fft = 4 * (n_spec-1) # for spectral interpolation
    rolloff_mp_lin_half = min_phase_half_spectrum(rolloff_lin_half,
                                                  n_fft=n_fft) #, half=False)
    # previous hack:
    # rolloff_mp_lin_whole = append_flip_conjugate(rolloff_mp_lin_half)
    # b_rolloff = np.fft.ifft(rolloff_mp_lin_whole)
    # new hack:
    b_rolloff = rolloff_mp_lin_half    
    a_rolloff = 1 # np.ones(1)
    n_b = 4
    n_a = 4
    label = f"{test_num}: 1/f^{power} rolloff filter, {n_a=} {n_b=}, n_spec {n_spec}"
    error1 = test_invfreqz(b_rolloff, a_rolloff, n_b, n_a,
                          n_spec, label, log_freq=True)
    error2 = test_invfreqz(b_rolloff, a_rolloff, n_b, n_a,
                           n_spec, label, log_freq=True, n_iter=5)
    print(f"---\nEquation-Error norm = {error1}")
    print(f"Output-Error norm = {error2}")
    return error1 + error2

if test_num == 33 or test_num == 0:
    power = 0.5
    total_error += model_incomplete_rolloff(power, test_num)

if test_num == 34 or test_num == 0:
    power = 1.5
    total_error += model_incomplete_rolloff(power, test_num)

# -------------------------------------------------------------------------------

if test_num == 0:
    print("--------------------------------------------------------------------------------")
    this_file = os.path.basename(__file__)
    print(f"{this_file}:\n Sum of all test errors is {total_error}")
