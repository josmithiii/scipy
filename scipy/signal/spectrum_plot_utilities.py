import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm


def dB(amplitude_array, clip=1e-8):
    """
    Convert real or complex numpy array to magnitude in decibels.

    Parameters:
    amplitude_array (numpy.ndarray): Input array of linear amplitudes (real or complex).
    clip (float): Minimum amplitude to consider (default: 1e-8).

    Returns:
    numpy.ndarray: Array of magnitudes in decibels.

    # Example usage
    if __name__ == "__main__":
        # Create a sample array
        sample_array = np.array([0, 1e-9, 1e-5, 0.1, 1, 10, 100])

        # Convert to dB
        db_result = dB(sample_array)

        # Print results
        for amp, db in zip(sample_array, db_result):
            print(f"Amplitude: {amp:.2e}, dB: {db:.2f}")
    """
    # Ensure the input is a numpy array
    amplitude_array = np.asarray(amplitude_array)

    # Calculate the magnitude of the array (handles both real and complex inputs)
    magnitude = np.abs(amplitude_array)

    # Clip values below the specified threshold
    clipped_magnitude = np.maximum(magnitude, clip)

    # Convert to decibels
    db_array = 20 * np.log10(clipped_magnitude)

    return db_array


def plot_mag_spectrum(mag_spec, wT=None, title=None, mag_units='dB'):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(mag_spec) if wT is None else plt.plot(wT,mag_spec)
        plt.title(title)
        plt.grid(True)
        plt.ylabel(f'Magnitude [{mag_units}]')
        plt.subplot(2, 1, 2)
        plt.semilogx(mag_spec) if wT is None else plt.semilogx(wT, mag_spec)
        plt.grid(True)
        plt.xlabel('Frequency [bins]')
        plt.ylabel(f'Magnitude [{mag_units}]')
        plt.tight_layout()
        plt.show()


def plot_signal(signal, title=None):
        plt.figure(figsize=(10, 6))
        plt.plot(signal)
        plt.title(title)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.xlabel('Time [samples]')
        plt.tight_layout()
        plt.show()


def plot_bode(w, H, title=None, save_path=None, display=True):
    # Convert frequency to Hz
    f = w / (2 * np.pi)

    # Calculate magnitude in dB
    db = 20 * np.log10(np.abs(H))

    # Create the Bode plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, db)
    # plt.plot(f, db)
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    if title is None:
        title = "Bode Plot"
    plt.title(title)
    plt.axhline(-3, color='green', linestyle=':', label='-3 dB Point')
    plt.legend()
    if save_path:
        plt.savefig(save_path, format='ps', dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}")
    if display:
        plt.show()
    else:
        plt.close()


def plot_bode_filter_responses(filter_responses, labels, cutoff,
                               save_path=None, display=True):
    plt.figure(figsize=(12, 8))

    for (w, h), label in zip(filter_responses, labels):
        db = 20 * np.log10(np.abs(h))
        plt.semilogx(w / (2 * np.pi), db, label=label)

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Bode Plot Comparison of Filters')
    plt.axvline(cutoff, color='red', linestyle='--', label='Cutoff Frequency')
    plt.axhline(-3, color='green', linestyle=':', label='-3 dB Point')

    plt.ylim(-80, 5)
    plt.xlim(0.1 * cutoff, 10 * cutoff)

    plt.legend()
    plt.show()

    if save_path:
        plt.savefig(save_path, format='ps', dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}")

    if display:
        plt.show()
    else:
        plt.close()


def plot_spectrum_overlay(spec1_lin, spec2_lin, w, title, lab1, lab2, log_freq=False):
    """Plot overlay of two complex spectra."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)

    # Limit the minimum magnitude to -80 dB
    min_db = -80
    spec1_db = 20 * np.log10(np.maximum(np.abs(spec1_lin), 10**(min_db/20)))
    spec2_db = 20 * np.log10(np.maximum(np.abs(spec2_lin), 10**(min_db/20)))

    # Plot Magnitude Response
    if log_freq:
        plt.semilogx(w, spec1_db, 'b', label=lab1)
        plt.semilogx(w, spec2_db, 'r--', label=lab2)
    else:
        plt.plot(w, spec1_db, 'b', label=lab1)
        plt.plot(w, spec2_db, 'r--', label=lab2)
    plt.title(f'{title} - Magnitude Response')
    plt.ylabel('Magnitude [dB]')
    max_db_plot = 1.1 * np.max(np.maximum(spec1_db, spec2_db))
    min_db_plot = np.min(np.minimum(spec1_db, spec2_db))
    plt.ylim(min_db_plot, max_db_plot)
    plt.legend()
    plt.grid(True)

    # Plot Phase Response
    spec1_lin[np.abs(spec1_lin) < 1.0e-12] = 0
    spec2_lin[np.abs(spec2_lin) < 1.0e-12] = 0
    plt.subplot(2, 1, 2)
    if log_freq:
        plt.semilogx(w, np.unwrap(np.angle(spec1_lin)), 'b', label=lab1)
        plt.semilogx(w, np.unwrap(np.angle(spec2_lin)), 'r--', label=lab2)
    else:
        plt.plot(w, np.unwrap(np.angle(spec1_lin)), 'b', label=lab1)
        plt.plot(w, np.unwrap(np.angle(spec2_lin)), 'r--', label=lab2)
    plt.title(f'{title} - Phase Response')
    plt.ylabel('Phase [rad]')
    plt.xlabel('Frequency [rad/sample]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return norm(spec1_lin - spec2_lin) / norm(spec1_lin)

