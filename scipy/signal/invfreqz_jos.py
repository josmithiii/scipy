"""
Module/script name: invfreqz_jos.py
Temporary test file for invfreqz proposal development.
Author: Julius Smith
Date: Started 9/03/24

Dependencies: See import and from below
Additional notes:
    Intended not to be included in the final scipy squash-merge,
    but rather absorbed only in final form into ./_filter_design.py

Background:
    Fast frequency-domain equation-error method for digital filter design

Created initially by pasting the LaTeX for
https://ccrma.stanford.edu/~jos/filters/FFT_Based_Equation_Error_Method.html
into Claude Sonnet 3.5, asking for a translation to Python, and
debugging the result on a number of simple tests.
The comments were also improved and extended.

@author: josmithiii@github

"""
import numpy as np
from scipy.linalg import toeplitz, solve, norm
from scipy.signal import freqz
from spectrum_utilities_jos import append_flip_conjugate
from filter_utilities_jos import check_roots_stability
from filter_plot_utilities_jos import plot_filter_analysis #, zplane
# from spectrum_plot_utilities_jos import plot_spectrum_overlay
from typing import Literal

# [B, A] = invfreqz(H, n_B, n_A, U, Wt, 
def invfreqz(
    H: np.ndarray,
    n_zeros: int,
    n_poles: int,
    U: np.ndarray   | None = None,
    weight: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    method: Literal['equation_error', 'prony', 'pade_prony'] = 'equation_error',
    method_iter: Literal['gauss_newton', 'steiglitz_mcbride'] = 'gauss_newton',
    n_iter: int     | None = 0,
    tol_iter: float | None = 1e-8,
    b_0: np.ndarray | None = None,
    a_0: np.ndarray | None = None,
    min_phase: bool | None = False,
    stabilize: bool | None = False,
    lr0: float      | None = 1.0,
    verbose: bool   | None = False,
    debug: bool     | None = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters ("opt" means "optional"):
        H (array): Desired frequency response, uniformly sampled,
                   including dc and pi, with no negative frequencies.
        n_zeros (int): Number of zeros in the filter.
        n_poles (int): Number of poles in the filter.
        U (array, opt): Input frequency response (equation-error weighting).
        weight (array, opt): output-error weigthing function.
        verbose (bool): Enables plotting and additional print statements.
        debug (bool): Enables plotting and additional print statements.
        n_iter (int, opt): Max number of iterations to use in method method_iter.
        method: 'equation_error' [default], 'prony', or 'pade_prony' [n_iter=0].
        method_iter: 'gauss_newton' [default] or 'steiglitz_mcbride' [n_iter>0].

    The following additional [optional] parameters only pertain to n_iter > 0:
        tol_iter (float, opt): Tolerance on the norm of the coefficients changes
                               at which to halt Steiglitz-McBride iterations.
        b_0 (array, opt): Initial numerator coefficients. [Zeros default]
        a_0 (array, opt): Initial denominator coefficients. Default is [1, zeros].
        min_phase (bool, opt): Convert H to minimum-phase first thing. Default is False.
        zero_clip (float): Threshold to avoid divide by 0 where needed. [1e-7]
        stabilize (bool): Reflect any unstable poles inside the unit circle.
                          [Default is False when n_iter is 0, else True]
        lr0 (float): Initial learning rate. Climbs from here to 1 over n_iter.
                     Setting to 1 to disables this feature.

    Returns:
        b (array): Numerator coefficients of the designed filter.
        a (array): Denominator coefficients of the designed filter.

    Note:
        For maximum efficiency, the number of frequency points (length of
        H and U) should be Nfft/2+1, where Nfft is a power of 2 (FFT size
        used herein).

    .. versionadded:: 1.14.2
    
    """

    if n_iter == 0:
        if method == 'prony':
            print ('return prony here')
        elif method == 'pade_prony':
            print ('return pade_prony here')
        else:
            if method != 'equation_error':
                print(f'*** invfreqz: unknown method "{method}" - '
                      'choosing equation_error')
            return fast_equation_error_filter_design(H, n_zeros, n_poles, U, omega,
                                                     debug=debug, verbose=verbose)
    else:
        if method_iter == 'steiglitz_mcbride':
            return fast_steiglitz_mcbride_filter_design(
                H, U, n_zeros, n_poles,
                n_iter=n_iter, tol_iter=tol_iter, b_0=None, a_0=None,
                zero_clip=1e-7, stabilize=stabilize, lr0=lr0,
                verbose=verbose, debug=debug )
        else:
            if method_iter != 'gauss_newton':
                print(f'*** invfreqz: unknown iterative method "{method_iter}" - '
                      'choosing gauss_newton')
            print ('return gauss_newton here')

def toeplitz_circulant_window(x, n_window):
    """
    Create an upper-left window of a Toeplitz circulant matrix from a
    given row vector and window size.

    Parameters:
    x (np.ndarray): Vector source of the first row of the matrix
    n_window (int): desired (square) matrix size

    Returns:
    numpy.ndarray: The Toeplitz circulant matrix

    Example usage:
    x = np.array([1, 2, 3, 4])
    result = toeplitz_circulant_window(x,3)
    print(result)  # Expect: [[1,2,3],[4,1,2],[3,4,1]]

    """
    n_max = len(x)
    assert n_max >= n_window, "requested window size exceeds given row vector"

    matrix = np.empty((n_window, n_window), dtype=x.dtype)

    for i in range(n_window):
        matrix[i] = np.roll(x, i)[:n_window]

    return matrix




def check_real(x: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Check that a complex numpy array is effectively real and return its real part.

    Parameters:
    x (np.ndarray): Input complex numpy array
    tol (float): Maximum warning-free ratio of imaginary-part-norm to array-norm

    Returns:
    numpy.ndarray: Real part of the input array

    Prints a warning if the norm of the imaginary part
    exceeds tol times the norm of the original array.
    """
    # Calculate the norms
    norm_original = norm(x)
    norm_imag = norm(x.imag)

    # Check if imaginary part is significant
    if norm_imag > tol * norm_original:
        print(f"Warning: Imaginary part norm ({norm_imag}) exceeds "
              f"{tol} times the original array norm ({norm_original})")

    return x.real


def fast_equation_error_filter_design(
    H: np.ndarray,
    n_zeros: int,
    n_poles: int,
    U: np.ndarray | None = None,
    omega: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Implements the fast equation-error filter design algorithm.

    Based on the algorithm described at:
    https://ccrma.stanford.edu/~jos/filters/FFT_Based_Equation_Error_Method.html

    Parameters:
    H: Desired frequency response, uniformly sampled, including dc and pi,
       with no negative frequencies
    n_zeros: Number of zeros in the filter
    n_poles: Number of poles in the filter
    U: Input signal frequency response (can be used for weighting)
    omega: Frequency grid. If None, a linear space from 0 to π is created

    Returns:
    tuple[np.ndarray, np.ndarray]: Coefficients of the designed filter
        in the order [numerator, denominator] of the transfer function

    Notes:
    For maximum efficiency, the number of frequency points
    (length of H and U) should be Nfft/2+1,
    where Nfft is a power of 2 (FFT size used herein).

    Raises:
    ValueError: If omega array doesn't cover either [0,pi] (for real filters)
                or [-pi,pi] (for complex filters).

    """

    N = len(H)  # power of 2 plus 1 most efficient

    if omega is None:
        omega = np.linspace(0, np.pi, N)
        is_complex = False
    else:
        is_complex = omega[0] < 0

    if not np.isclose(omega[-1], np.pi):
        raise ValueError(f"""
        The last element of omega ({omega[-1]})
        must be approximately pi ({np.pi})
        """)

    if is_complex and not np.isclose(omega[0], -np.pi):
        raise ValueError(f"""
        The first element of omega ({omega[0]}) must be approximately
        -pi ({-np.pi}) for complex-filter design
        """)

    if U is None:  # Make it effectively all 1s (no frequency-weighting == "impulse")
        Y = H
        Y = append_flip_conjugate(Y)
        r_uu = np.zeros_like(H)
        r_uu[0] = 1
        r_yu = check_real(np.fft.ifft(Y))  # alias_N(impulse_response)
    else:
        Y = H * U  # Filter output spectrum Y when input spectrum is U
        Y = append_flip_conjugate(Y)
        U = append_flip_conjugate(U)  # Append (-) frequencies for complete spectrum
        r_uu = np.fft.ifft(np.abs(U)**2)  # autocorrelation of input signal, aliased_N
        r_yu = check_real(np.fft.ifft(Y * np.conj(U)))  # input-output cross-correlation
                                                        # == alias_N(impulse_response)

    r_yy = np.fft.ifft(np.abs(Y)**2)

    # Construct Toeplitz matrices
    R_yy = toeplitz_circulant_window(r_yy, n_poles)
    R_uu = toeplitz_circulant_window(r_uu, n_zeros + 1)
    col1 = np.roll(r_yu, 1)[:n_zeros + 1] # r_yu[N-1:N-n_poles-1:-1]
    Nr = len(r_yu)
    row1 = np.flip(r_yu[Nr - n_poles:Nr])
    R_yu = toeplitz(col1, row1)
    R_uy = np.conj(R_yu.T)

    # Construct the block matrix and the right-hand side vector
    A = np.block([[R_uu, R_yu], [R_uy, R_yy]])
    b = np.concatenate([r_yu[:n_zeros + 1], r_yy[1:n_poles + 1]])

    # Solve the system of equations
    x = solve(A, b)

    # Extract the filter coefficients
    b_coeffs = x[:n_zeros + 1]
    a_coeffs = np.concatenate([[1.0], -x[n_zeros + 1:]])

    return check_real(b_coeffs), check_real(a_coeffs)


def clipped_magnitude_array_inverse(A, zero_clip=1e-7):
    A_array = np.asarray(A)
    return np.reciprocal(np.maximum(zero_clip, A_array))


def clipped_real_array_inverse(A, zero_clip=1e-7):
    A_array = np.asarray(A)
    magnitude = np.abs(A_array)
    clipped_magnitude = np.maximum(zero_clip, magnitude)
    return np.sign(A_array) / clipped_magnitude


def invert_unstable_roots(A):
    print(f"invert_unstable_roots: input is {A}")
    roots = np.roots(A)
    unstable_mask = np.abs(roots) > 1
    if not np.any(unstable_mask):
        return A, roots, True  # All roots are stable
    # Invert the unstable roots:
    roots[unstable_mask] = 1 / np.conj(roots[unstable_mask])
    A_stable = np.poly(roots) # reconstruct the polynomial coefficients
    A_stable = check_real(A_stable)
    return A_stable, roots, False  # Some roots were unstable and inverted


def exp_window(A, r):
    """
    Apply pointwise exponential window [1, r, r^2, ...]
    to the elements of 1D numpy array A.

    Parameters:
    A (numpy.ndarray): 1D input array
    r (float): Base of the exponential window

    Returns:
    numpy.ndarray: Array A with exponential window applied
    """
    # Check if A is 1D
    if A.ndim != 1:
        raise ValueError("Input array A must be 1-dimensional")

    # Create the exponential window
    window = r ** np.arange(len(A))

    # Apply the window to A
    return A * window


def fast_steiglitz_mcbride_filter_design(H, U, n_zeros, n_poles, n_iter=5,
                                         tol_iter=1e-8, b_0=None, a_0=None,
                                         zero_clip=1e-7, stabilize=True,
                                         lr0=1,
                                         debug=True, verbose=True):
    """Frequency-domain Steiglitz-McBride algorithm.

    The Steiglitz-McBride algorithm converts an equation-error filter
    design to an output-error filter design.  To accomplish this, it
    iteratively calls `fast_equation_error_filter_design`, applying the
    filter 1/a to both input and output on each iteration until either
    the maximum number of iterations is reached or the stopping
    tolerance in successive filter changes is achieved.

    Parameters:
    H (array): Desired frequency response, uniformly sampled,
               including dc and pi, with no negative frequencies.
    U (array): Input frequency response (can be used for weighting).
    n_zeros (int): Number of zeros in the filter.
    n_poles (int): Number of poles in the filter.
    n_iter (int): Max number of iterations of the Steiglitz-McBride algorithm.
    tol_iter (float): Tolerance on the norm of the coefficients changes
                                  at which to halt Steiglitz-McBride iterations.
    b_0 (array, optional): Initial numerator coefficients. Default is zeros.
    a_0 (array, optional): Initial denominator coefficients. Default is [1, zeros].
    zero_clip (float): Threshold to avoid divide by zero in frequency response inverse.
                       Default is 1e-7.
    stabilize (bool): When true, reflect any unstable poles
                      inside the unit circle if they go unstable.
    lr0 (float): learning rate climbs from here to 1
                      over n_iter. Set to 1 to disable this feature.
    debug (bool): Enables plotting and additional print statements.
    verbose (bool): Prints convergence progress each iteration.

    Returns:
    b (array): Numerator coefficients of the designed filter.
    a (array): Denominator coefficients of the designed filter.

    For maximum efficiency, the number of frequency points (length of
    H and U) should be Nfft/2+1, where Nfft is a power of 2 (FFT size
    used herein).

    """

    # Initialize filter coefficients
    current_b = b_0 if b_0 is not None else np.zeros(n_zeros + 1)
    current_a = a_0 if a_0 is not None else np.hstack((1, np.zeros(n_poles)))
    iterations = 0

    N = len(H)
    w = np.linspace(0, np.pi, N)

    # If U is None, default to an array of ones (no weighting)
    if U is None:
        U = np.ones_like(H)

    # Initialize H_local and U_local with copies of H and U
    H_local = H.copy()
    U_local = U.copy()

    learning_rate = lr0
    delta_learning_rate = (1.0 - lr0) / n_iter

    initial_coeffs_used = False

    while True:
        print(f"\n------- iteration {iterations} -----------")

        if not initial_coeffs_used and b_0 is not None and a_0 is not None:
            new_b = b_0
            new_a = a_0
            initial_coeffs_used = True
        else:
            # Perform equation error filter design
            try:
                new_b, new_a = fast_equation_error_filter_design(
                    H_local, n_zeros, n_poles, U=U_local, omega=w)
            except np.linalg.LinAlgError as e:
                raise ValueError("Linear algebra error during "
                                 f"iteration {iterations}: {e}")

        print(f"{new_b = }")
        print(f"{new_a = }")

        # Stabilize the filter if required
        if stabilize:
            new_a, _, _ = invert_unstable_roots(new_a)
            if debug:
                print(f"After inverting unstable roots, {new_a=}")

        # Compute the norm of the change in coefficients
        if debug:
            # freqz(new_b, new_a)
            # zplane(new_b, new_a, title)
            title = f"Steiglitz-McBride Iteration {iterations}"
            error_freq_resp = plot_filter_analysis(H, 1, new_b, new_a, w, title,
                                                   show_plot=True, log_freq=True)
            print(f"norm(frequency_response_error) = {error_freq_resp}")

        norm_change = norm(new_a - current_a) + norm(new_b - current_b)
        if debug or verbose:
            print(f"norm_change in a at iteration {iterations}: {norm_change}")

        # Check for convergence
        if norm_change < tol_iter * norm(current_a):
            if debug or verbose:
                print(f"""
                Stopping tolerance {tol_iter} reached
                after {iterations + 1} iterations.""")
            break
        if iterations >= n_iter:
            if debug or verbose:
                print(f"Reached maximum of {iterations} iterations.")
            break

        # Update current coefficients
        current_a = new_a
        current_b = new_b
        iterations += 1

        # Compute the inverse frequency response of the current denominator polynomial
        # Ai = clipped_real_array_inverse(A, zero_clip)
        # Compute the inverse frequency response of the current denominator polynomial
        if learning_rate < 1.0:
            windowed_a = exp_window(current_a, learning_rate)
            wA, Ai = freqz([1], windowed_a, worN=N)
            learning_rate += delta_learning_rate
        else:
            wA, Ai = freqz([1], current_a, worN=N)  # 1 / A(z)

        # if debug:
        #     A = np.reciprocal(Ai)
        #     plot_spectrum_overlay(A, Ai, wA, "A and 1/A", "A", "1/A")

        # Update H_local and U_local using the original H and U and new A inverse:
        H_local = H * Ai
        U_local = U * Ai

        if debug:
            title = f"Steiglitz-McBride Iteration FINAL, after {iterations} iterations"
            error_freq_resp = plot_filter_analysis(H, 1, new_b, new_a, w, title,
                                                   show_plot=True, log_freq=True)
            # _, Hh = freqz(new_b, new_a, worN=w)
            # title = f"Steiglitz-McBride Iteration {iterations}"
            # error_freq_resp = plot_spectrum_overlay(H, Hh, w / np.pi, title, "
            # f"Desired, Iteration {iterations}",
            # log_freq=False)
            print(f"{title}: norm(frequency_response_err) = {error_freq_resp}")

    return new_b, new_a


# Example usage:
if __name__ == "__main__":
    from scipy.signal import freqz

    # import pdb
    # pdb.set_trace()

    N = 1024 # power of 2 preferred
    title = "Pathological test example for regression testing only"
    N = 64 # multiples of 3 yield singularities
           # which could be handled symbolically (isolated NaNs are clear poles)
    b = [1, 2, 3, 2, 3]
    a = [1, 2, 3, 2, 1, 4]
    n_b = len(b)-1  # number of zeros
    n_a = len(a)-1  # number of poles
    model_complete_case = True  # In the model-complete case,
                                # the filter design can exactly match the desired
    if not model_complete_case:
        n_b = 1 # one too few zeros
        n_a = 1 # one too few poles
    w = np.linspace(0, np.pi, int(N+1))
    _,H = freqz(b, a, worN=w)
    U = np.ones_like(H)
    bh, ah = invfreqz(H, n_b, n_a, U=U)
    print(f"\n{title}:")
    print("Original coefficients:")
    print(f"b = {b}")
    print(f"a = {a}")
    print("Estimated coefficients:")
    print(f"bh = {bh}")
    print(f"ah = {ah}")
    if model_complete_case:
        print("Errors:")
        print(f"b-bh = {b-bh}")
        print(f"a-ah = {a-ah}")
        print("Total Error:")
        print(f"norm(a-ah) + norm(b-ba) = {norm(a-ah) + norm(b-bh)}")
    print("--------------------------------------------------------------")
    print("Steiglitz McBride:")
    bh, ah = fast_steiglitz_mcbride_filter_design(H, U, n_b, n_a,
                                                  n_iter=30,
                                                  tol_iter=1e-8,
                                                  b_0=bh, a_0=ah, lr0=1,
                                                  debug=False )
    print(f"\n{title}:")
    print("Original coefficients:")
    print(f"b = {b}")
    print(f"a = {a}")
    print("Estimated coefficients:")
    print(f"bh = {bh}")
    print(f"ah = {ah}")
    if model_complete_case:
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
        num_marginally_stable = np.sum((magnitudes >= 1.0 - tol) & \
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
