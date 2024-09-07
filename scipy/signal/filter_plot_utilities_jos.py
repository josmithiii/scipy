import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

def plot_spectrum_overlay(spec1, spec2, w, title, lab1, lab2):
    """Plot overlay of two spectra."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)

    # Limit the minimum magnitude to -80 dB
    min_db = -80
    spec1_db = 20 * np.log10(np.maximum(np.abs(spec1), 10**(min_db/20)))
    spec2_db = 20 * np.log10(np.maximum(np.abs(spec2), 10**(min_db/20)))

    # Plot Magnitude Response
    #plt.semilogx(w, spec1_db, 'b', label=lab1)
    #plt.semilogx(w, spec2_db, 'r--', label=lab2')
    plt.plot(w, spec1_db, 'b', label=lab1)
    plt.plot(w, spec2_db, 'r--', label=lab2)
    plt.title(f'{title} - Magnitude Response')
    plt.ylabel('Magnitude [dB]')
    plt.ylim(min_db, 5)  # Set y-axis limits
    plt.legend()
    plt.grid(True)

    # Plot Phase Response
    spec1[np.abs(spec1) < 1.0e-12] = 0
    spec2[np.abs(spec2) < 1.0e-12] = 0
    plt.subplot(2, 1, 2)
    #plt.semilogx(w, np.unwrap(np.angle(spec1)), 'b', label=lab1)
    #plt.semilogx(w, np.unwrap(np.angle(spec2)), 'r--', label=lab2)
    plt.plot(w, np.unwrap(np.angle(spec1)), 'b', label=lab1)
    plt.plot(w, np.unwrap(np.angle(spec2)), 'r--', label=lab2)
    plt.title(f'{title} - Phase Response')
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [rad/sample]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_frequency_response_fit(b_orig, a_orig, b_est, a_est, w, title):
    """Plot frequency-response fit of original and estimated filters."""
    wo, h_orig = freqz(b_orig, a_orig, worN=w)
    we, h_est = freqz(b_est, a_est, worN=w)
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

    # Plot Magnitude Response
    #plt.semilogx(w, h_orig_db, 'b', label='Original')
    #plt.semilogx(w, h_est_db, 'r--', label='Estimated')
    plt.plot(w, h_orig_db, 'b', label='Original')
    plt.plot(w, h_est_db, 'r--', label='Estimated')
    plt.title(f'{title} - Magnitude Response')
    plt.ylabel('Magnitude [dB]')
    plt.ylim(min_db, 5)  # Set y-axis limits
    plt.legend()
    plt.grid(True)

    # Plot Phase Response
    h_orig[np.abs(h_orig) < 1.0e-12] = 0
    h_est[np.abs(h_est) < 1.0e-12] = 0
    plt.subplot(2, 1, 2)
    #plt.semilogx(w, np.unwrap(np.angle(h_orig)), 'b', label='Original')
    #plt.semilogx(w, np.unwrap(np.angle(h_est)), 'r--', label='Estimated')
    plt.plot(w, np.unwrap(np.angle(h_orig)), 'b', label='Original')
    plt.plot(w, np.unwrap(np.angle(h_est)), 'r--', label='Estimated')
    plt.title(f'{title} - Phase Response')
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [rad/sample]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def zplane(b, a):
    """Plot the complex z-plane given a transfer function."""
    
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
    ax.set_title('Pole-Zero Plot')
    ax.legend()
    
    # Add grid
    ax.grid(True)
    
    # Show the plot
    plt.show()

# Example usage
b = [1, -0.5, 0.25]  # Numerator coefficients
a = [1, -0.7, 0.1]   # Denominator coefficients

zplane(b, a)
