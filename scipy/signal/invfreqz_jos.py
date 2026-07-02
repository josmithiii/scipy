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
@license: BSD 3-Clause "New" or "Revised" License: https://github.com/scipy/scipy/blob/main/LICENSE.txt

"""
import numpy as np
from scipy.linalg import toeplitz, lstsq, norm
from scipy.signal import freqz
from spectrum_utilities_jos import append_flip_conjugate, min_phase_half_spectrum
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
    method_iter: Literal['gauss_newton', 'steiglitz_mcbride'] = 'steiglitz_mcbride',
    n_iter: int     = 0,
    tol_iter: float | None = 1e-8,
    b_0: np.ndarray | None = None,
    a_0: np.ndarray | None = None,
    zero_clip: float | None = None,
    min_phase: bool | None = False,
    stabilize: bool | None = None,
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
        method_iter: 'steiglitz_mcbride' [default] or 'gauss_newton' [n_iter>0].
                     (Only 'steiglitz_mcbride' is currently implemented.)
        min_phase (bool, opt): Convert H to minimum-phase first thing. Default is False.
        stabilize (bool): Reflect any unstable poles inside the unit circle.
                          [Default is False when n_iter is 0, else True]

    The following additional [optional] parameters only pertain to n_iter > 0:
        tol_iter (float, opt): Tolerance on the norm of the coefficients changes
                               at which to halt Steiglitz-McBride iterations.
        b_0 (array, opt): Initial numerator coefficients. [Zeros default]
        a_0 (array, opt): Initial denominator coefficients. Default is [1, zeros].
        zero_clip (float): NOT YET IMPLEMENTED (must be None). Intended as a
                           threshold to avoid divide by 0 where needed.
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

    # ---- Validate all arguments up front, regardless of the branch taken ----

    if not isinstance(n_iter, int) or n_iter < 0:
        raise ValueError(
            f"invfreqz: n_iter must be a non-negative int, got {n_iter!r}")

    if method not in ('equation_error', 'prony', 'pade_prony'):
        raise ValueError(f"invfreqz: unknown method {method!r}")
    if method_iter not in ('gauss_newton', 'steiglitz_mcbride'):
        raise ValueError(f"invfreqz: unknown iterative method {method_iter!r}")

    if method in ('prony', 'pade_prony'):
        raise NotImplementedError(
            f"invfreqz: method={method!r} is not yet implemented")
    if n_iter > 0 and method_iter == 'gauss_newton':
        raise NotImplementedError(
            "invfreqz: method_iter='gauss_newton' is not yet implemented")

    if weight is not None:
        raise NotImplementedError(
            "invfreqz: output-error 'weight' argument is not yet implemented")
    if zero_clip is not None:
        raise NotImplementedError(
            "invfreqz: zero_clip is accepted for API compatibility but is not "
            "used by any implemented method; pass None")
    if n_iter > 0 and omega is not None:
        raise NotImplementedError(
            "invfreqz: a custom omega grid is not yet supported on the "
            "iterative (Steiglitz-McBride) path, which uses "
            "linspace(0, pi, len(H))")

    if min_phase:
        if omega is not None and omega[0] < 0:
            raise NotImplementedError(
                "invfreqz: min_phase=True requires a half-spectrum H "
                "(omega from dc to pi inclusive); whole-spectrum "
                "(negative-frequency) input is not supported")
        n_fft_mp = 4 * (len(H) - 1)
        try:
            H = min_phase_half_spectrum(H, n_fft_mp)
        except ValueError as e:
            raise ValueError(
                "invfreqz: min-phase conversion of H failed at the internal "
                f"FFT size n_fft={n_fft_mp}. Smooth H, or convert it yourself "
                "via min_phase_half_spectrum(H, n_fft) with a larger n_fft "
                "and call invfreqz with min_phase=False.") from e

    # Documented default: no stabilization for the direct solve (n_iter == 0),
    # stabilization on for the iterative methods (n_iter > 0).
    if stabilize is None:
        stabilize = n_iter != 0

    if n_iter == 0:
        b, a = fast_equation_error_filter_design(H, n_zeros, n_poles, U, omega)
        if stabilize:
            a, _, was_stable = invert_unstable_roots(a)
            if not was_stable:
                b, a = b / a[0], a / a[0]  # restore a[0] == 1
            if debug:
                print(f"After inverting unstable roots, {a=}")
        return b, a
    else:
        return fast_steiglitz_mcbride_filter_design(
            H, U, n_zeros, n_poles,
            n_iter=n_iter, tol_iter=tol_iter, b_0=b_0, a_0=a_0,
            stabilize=stabilize, lr0=lr0,
            verbose=verbose, debug=debug)

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

    if n_poles == 0:
        # FIR (all-zero) fit: solve R_uu @ b = r_yu[:n_zeros+1] alone.  (The
        # block construction below cannot form its zero-size R_yy/R_yu
        # blocks.)  With U=None this reduces to impulse-response truncation.
        R_uu = toeplitz_circulant_window(check_real(r_uu), n_zeros + 1)
        b_fir, *_ = lstsq(R_uu, r_yu[:n_zeros + 1])
        return check_real(b_fir), np.ones(1)

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

    # Solve the system of equations with SVD-based lstsq rather than solve():
    # for near-allpass targets (|H| ~ const, e.g. the moForte string
    # loop-filter R(z)) the normal-equation block matrix is severely
    # ill-conditioned (rcond ~ 1e-18) and plain solve() emits LinAlgWarning
    # and amplifies noise, while SVD degrades gracefully.  Decision recorded
    # 2026-07; matches the upstream PR (see TODO_INVFREQZ.md section 2).
    x, *_ = lstsq(A, b)

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
    """Reflect any roots of A outside the unit circle to their conjugate
    reciprocals, scaling the result so the magnitude response is unchanged:
    |A_stable(e^jw)| == |A(e^jw)| for all w (each reflected root r scales the
    factor magnitude by 1/|r|, so the polynomial is multiplied by prod |r|).
    Note A_stable is then NOT monic (A_stable[0] = A[0] * prod |r|); when a
    monic denominator is needed, divide (b, a) through by A_stable[0] to
    preserve the transfer function's magnitude.

    Returns (A_stable, roots_after_inversion, was_stable)."""
    roots = np.roots(A)
    unstable_mask = np.abs(roots) > 1
    if not np.any(unstable_mask):
        return A, roots, True  # All roots are stable
    # Invert the unstable roots, compensating the gain:
    gain = np.prod(np.abs(roots[unstable_mask]))
    roots[unstable_mask] = 1 / np.conj(roots[unstable_mask])
    A_stable = A[0] * gain * np.poly(roots)  # reconstruct the coefficients
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
                                         stabilize=True,
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
    current_b = np.asarray(b_0, dtype=float) if b_0 is not None \
        else np.zeros(n_zeros + 1)
    current_a = np.asarray(a_0, dtype=float) if a_0 is not None \
        else np.hstack((1, np.zeros(n_poles)))
    iterations = 0

    N = len(H)
    w = np.linspace(0, np.pi, N)

    # If U is None, default to an array of ones (no weighting)
    if U is None:
        U = np.ones_like(H)

    # Steiglitz-McBride prefilters the output signal spectrum Y = H*U and the
    # input spectrum U by 1/A of the previous iterate.  In the (H, U)
    # parameterization used by fast_equation_error_filter_design, that means
    # U_local = U/A with H UNCHANGED, since H = Y'/U' is invariant under
    # common prefiltering.  (Multiplying H by 1/A as well -- the bug fixed in
    # the upstream port -- applies 1/A twice to the output side, biasing
    # every iteration; it made model-complete iterates drift away from the
    # exact solution.)
    U_local = U.copy()

    # Warm start: apply the initial guess as the first 1/A_0 prefilter so the
    # first equation-error solve benefits from it.  (Do NOT feed b_0/a_0
    # through the loop as if they were a solve result: with current == new the
    # convergence test sees norm_change == 0 and returns the initial guess
    # unchanged, without ever running a design iteration.)
    if a_0 is not None:
        _, Ai0 = freqz([1], current_a, worN=w)  # 1 / A_0(z)
        U_local = U * Ai0

    learning_rate = lr0
    delta_learning_rate = (1.0 - lr0) / n_iter

    while True:
        print(f"\n------- iteration {iterations} -----------")

        # Perform equation error filter design
        try:
            new_b, new_a = fast_equation_error_filter_design(
                H, n_zeros, n_poles, U=U_local, omega=w)
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

        # Compute the inverse frequency response of the current denominator.
        # Evaluate 1/A on the SAME grid w (dc..pi inclusive) that H and U use.
        # Passing an integer worN=N would use freqz's endpoint-exclusive grid
        # linspace(0, pi, N, endpoint=False), misaligning the prefilter by up
        # to one bin (worst near Nyquist) and biasing every iteration.
        if learning_rate < 1.0:
            windowed_a = exp_window(current_a, learning_rate)
            wA, Ai = freqz([1], windowed_a, worN=w)
            learning_rate += delta_learning_rate
        else:
            wA, Ai = freqz([1], current_a, worN=w)  # 1 / A(z)

        # if debug:
        #     A = np.reciprocal(Ai)
        #     plot_spectrum_overlay(A, Ai, wA, "A and 1/A", "A", "1/A")

        # SM update: prefilter the input spectrum only, always from the
        # original U (the prefilter is 1/A_current, not cumulative); H stays
        # unchanged -- see the note above the loop.
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

    # In-loop stabilization gain-compensates a (non-monic); restore a[0] == 1
    # without changing the transfer function.
    if new_a[0] != 1.0:
        new_b, new_a = new_b / new_a[0], new_a / new_a[0]
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
    # Imported here (not at module top) to avoid a circular import:
    # filter_test_utilities_jos imports from this module.
    from filter_test_utilities_jos import report_stability
    report_stability(ah, a)
