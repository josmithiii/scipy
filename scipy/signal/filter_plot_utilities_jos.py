"""
Module/script name: filter_plot_utilities_jos.py
Temporary file for invfreqz proposal development.
Author: Julius Smith
Date: Started 9/03/24

Additional notes:
    Intended not to be included in the final scipy pull-request squash-merge,
    but rather adapted into scipy unit tests which I've not yet learned about.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.linalg import norm


# This crock is a direct result of letting Claude 3.5 write my initial tests.
# There was nothing wrong with what it did, but it chose limiting assumptions
# that I am now working around.  Maybe the prompt should emphasize general
# utility.
def get_freq_response(b, a, n_spec):
    have_truth = True
    if (len(b) == n_spec or len(b) == 2*(n_spec-1)) and np.isscalar(a) and a == 1:
        print("get_freq_response: Assuming b is the DESIRED SPECTRUM")
        have_truth = False
        
    w = np.linspace(0, np.pi, int(n_spec))
    if have_truth:
        w2,H = freqz(b, a, worN=w)
        if (not np.allclose(w,w2)):
            print("*** freqz is not computing frequencies as expected")
    else:
        w2 = w
        H = b

    return w2, H, have_truth


# def plot_frequency_response(b, a, w, title):

def plot_frequency_response_fit(b_orig, a_orig, b_est, a_est, w, title,
                                show_plot=False, log_freq=False):
    """Plot frequency-response fit of original and estimated filters."""
    n_spec = len(w)
    wo, h_orig, have_truth = get_freq_response(b_orig, a_orig, n_spec)
    # freqz(b_orig, a_orig, worN=w)
    we, h_est = freqz(b_est, a_est, worN=w)

    norm_of_difference = norm(h_orig - h_est) / norm(h_orig)

    if (not np.allclose(w, wo, atol=1e-12)):
        print("*** plot_frequency_response_fit: freqz changed original axis")
    if (not np.allclose(w, we, atol=1e-12)):
        print("*** plot_frequency_response_fit: freqz changed estimate axis")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)

    # Limit the minimum magnitude to -80 dB
    min_db = -80
    h_orig_db = 20 * np.log10(np.maximum(np.abs(h_orig), 10**(min_db/20)))
    h_est_db = 20 * np.log10(np.maximum(np.abs(h_est), 10**(min_db/20)))
    max_db = np.max(np.maximum(h_orig_db, h_est_db))
    min_db_plot = np.min(np.minimum(h_orig_db, h_est_db))
    max_db_plot = 1.1 * 5 * math.ceil(max_db / 5)

    # Plot Magnitude Response
    if log_freq:
        plt.semilogx(w, h_orig_db, 'b', label='Original')
        plt.semilogx(w, h_est_db, 'r--', label='Estimated')
    else:
        plt.plot(w, h_orig_db, 'b', label='Original')
        plt.plot(w, h_est_db, 'r--', label='Estimated')
    plt.title(f'{title} - Magnitude Response')
    plt.ylabel('Magnitude [dB]')
    plt.ylim(min_db_plot, max_db_plot)  # Set y-axis limits
    plt.legend()
    plt.grid(True)

    # Plot Phase Response
    h_orig[np.abs(h_orig) < 1.0e-12] = 0
    h_est[np.abs(h_est) < 1.0e-12] = 0
    plt.subplot(2, 1, 2)
    if log_freq:
        plt.semilogx(w, np.unwrap(np.angle(h_orig)), 'b', label='Original')
        plt.semilogx(w, np.unwrap(np.angle(h_est)), 'r--', label='Estimated')
    else:
        plt.plot(w, np.unwrap(np.angle(h_orig)), 'b', label='Original')
        plt.plot(w, np.unwrap(np.angle(h_est)), 'r--', label='Estimated')
    plt.title(f'{title} - Phase Response')
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [rad/sample]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if show_plot:
        plt.show()

    return norm_of_difference


def splane(b, a, title="Pole-Zero Plot in the s Plane"):
    # Plotting the Poles and Zeros in the s-plane
    plt.figure(figsize=(8, 6))
    s_poles = np.roots(a)
    s_zeros = np.roots(b)
    plt.plot(np.real(s_poles), np.imag(s_poles), 'bx', label='Poles')
    plt.plot(np.real(s_zeros), np.imag(s_zeros), 'ro', label='Zeros')
    plt.title(title)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def zplane(b, a, title="Pole-Zero Plot in the z Plane"):
    """
    Plot the complex z-plane given a transfer function.

    # Example usage
    b = [1, -0.5, 0.25]  # Numerator coefficients
    a = [1, -0.7, 0.1]   # Denominator coefficients

    zplane(b, a)

    """

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Plot the unit circle
    unit_circle = plt.Circle((0,0), 1, fill=False, color='gray', ls='dashed')
    ax.add_patch(unit_circle)

    # Plot the zeros
    ax.scatter(z.real, z.imag, marker='o', s=100, color='b', label='Zeros')

    # Plot the poles
    ax.scatter(p.real, p.imag, marker='x', s=100, color='r', label='Poles')

    # Set the limits
    r = 1.5 * max(np.max(np.abs(z)), np.max(np.abs(p)), 1)
    ax.set_xlim((-r, r))
    ax.set_ylim((-r, r))

    # Make the plot square
    ax.set_aspect('equal', adjustable='box')

    # Add labels and title
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title(title)
    ax.legend()

    # Add grid
    ax.grid(True)

    # Show the plot
    plt.show()


def plot_filter_analysis(b_orig, a_orig, b_est, a_est, w, title,
                         show_plot=True, log_freq=False):
    """
    Plot frequency-response fit of original and estimated filters,
    along with the pole-zero plot for both filters.
    """
    fig = plt.figure(figsize=(18, 10))
    
    # Frequency response plots
    n_spec = len(w)
    wo, H_orig, have_truth = get_freq_response(b_orig, a_orig, n_spec)
    we, H_est = freqz(b_est, a_est, worN=wo)

    # Magnitude Response
    ax1 = fig.add_subplot(221)
    min_db = -80
    H_orig_db = 20 * np.log10(np.maximum(np.abs(H_orig), 10**(min_db/20)))
    H_est_db = 20 * np.log10(np.maximum(np.abs(H_est), 10**(min_db/20)))
    
    if log_freq:
        ax1.semilogx(w, H_orig_db, 'b', label='Original')
        ax1.semilogx(w, H_est_db, 'r--', label='Estimated')
    else:
        ax1.plot(w, H_orig_db, 'b', label='Original')
        ax1.plot(w, H_est_db, 'r--', label='Estimated')
    
    ax1.set_title(f'{title} - Magnitude Response')
    ax1.set_ylabel('Magnitude [dB]')
    ax1.legend()
    ax1.grid(True)
    
    # Phase Response
    p_orig = np.unwrap(np.angle(H_orig))
    mask_orig = H_orig_db <= min_db
    p_orig[mask_orig] = 0

    p_est = np.unwrap(np.angle(H_est))
    mask_est = H_est_db <= min_db
    p_est[mask_est] = 0

    ax2 = fig.add_subplot(223)
    if log_freq:
        ax2.semilogx(w, p_orig, 'b',   label='Original')
        ax2.semilogx(w, p_est,  'r--', label='Estimated')
    else:
        ax2.plot(w, p_orig, 'b',   label='Original')
        ax2.plot(w, p_est,  'r--', label='Estimated')
    
    ax2.set_title(f'{title} - Phase Response')
    ax2.set_ylabel('Phase [rad]')
    ax2.set_xlabel('Frequency [rad/sample]')
    ax2.legend()
    ax2.grid(True)
    
    # Pole-Zero plots
    def plot_pz(b, a, ax, label):
        p = np.zeros(1) if np.isscalar(a) else np.roots(a)
            # Actually there are len(b)-1) zeros, but let's not plot them all
        z = np.roots(b)
        
        # Plot the unit circle
        unit_circle = plt.Circle((0,0), 1, fill=False, color='gray', ls='dashed')
        ax.add_patch(unit_circle)
        
        # Plot zeros and poles
        ax.scatter(z.real, z.imag, marker='o', s=100, color='b', label=f'{label} Zeros')
        ax.scatter(p.real, p.imag, marker='x', s=100, color='r', label=f'{label} Poles')
        
        # Set the limits
        r = 1.5 * max(np.max(np.abs(z)), np.max(np.abs(p)), 1)
        ax.set_xlim((-r, r))
        ax.set_ylim((-r, r))
        
        # Make the plot square
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.legend()
        ax.grid(True)
    
    if have_truth:
        # Original filter pole-zero plot
        ax3 = fig.add_subplot(222)
        plot_pz(b_orig, a_orig, ax3, "Original")
        ax3.set_title(f'{title} - Original Filter Pole-Zero Plot')
    
    # Estimated filter pole-zero plot
    ax4 = fig.add_subplot(224)
    plot_pz(b_est, a_est, ax4, "Estimated")
    ax4.set_title(f'{title} - Estimated Filter Pole-Zero Plot')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    # Replot frequency responses (no pole-zero plots)
    # over both linear and log frequencies

    if log_freq: # but skip this if log_freq not chosen

        fig = plt.figure(figsize=(18, 10))

        ax11 = fig.add_subplot(221)
        ax11.semilogx(w, H_orig_db, 'b', label='Original')
        ax11.semilogx(w, H_est_db, 'r--', label='Estimated')
        ax11.set_ylabel('Magnitude [dB]')
        ax11.legend()
        ax11.grid(True)
        ax11.set_title(f'{title}')

        ax12 = fig.add_subplot(222)
        ax12.plot(w, H_orig_db, 'b', label='Original')
        ax12.plot(w, H_est_db, 'r--', label='Estimated')
        ax12.set_ylabel('Magnitude [dB]')
        ax12.legend()
        ax12.grid(True)
        ax12.set_title(f'{title}')

        ax21 = fig.add_subplot(223)
        ax21.semilogx(w, p_orig, 'b',   label='Original')
        ax21.semilogx(w, p_est,  'r--', label='Estimated')
        ax21.set_ylabel('Phase [rad]')
        ax21.legend()
        ax21.grid(True)

        ax22 = fig.add_subplot(224)
        ax22.plot(w, p_orig, 'b',   label='Original')
        ax22.plot(w, p_est,  'r--', label='Estimated')
        ax22.set_ylabel('Phase [rad]')
        ax22.legend()
        ax22.grid(True)

        ax11.set_xlabel('Frequency [rad/sample]')
        ax12.set_xlabel('Frequency [rad/sample]')
        ax21.set_xlabel('Frequency [rad/sample]')
        ax22.set_xlabel('Frequency [rad/sample]')

        if show_plot:
            plt.show()

    error_freq_resp = norm(H_orig - H_est)
    return error_freq_resp

# Example usage
if __name__ == "__main__":
    # Example filter coefficients
    b_orig = [1, -0.5, 0.25]
    a_orig = [1, -0.7, 0.1]
    b_est = [0.9, -0.45, 0.2]
    a_est = [1, -0.65, 0.08]
    
    # Generate frequency points
    w = np.linspace(0, np.pi, 1000)
    
    # Call the function
    plot_filter_analysis(b_orig, a_orig, b_est, a_est, w, "Filter Analysis Example")    
