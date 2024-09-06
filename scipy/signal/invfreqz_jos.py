"""Fast frequency-domain equation-error method for digital filter design

Started 9/03/24 by pasting the LaTeX for
https://ccrma.stanford.edu/~jos/filters/FFT_Based_Equation_Error_Method.html
into Claude Sonnet 3.5, asking for a translation to Python, and
debugging the result on a number of simple tests.
Comments also improved and extended.

Maybe add this to scipy/signal/_filter_design.py

@author: josmithiii@github

"""
import numpy as np
from scipy.linalg import toeplitz, solve, norm
from scipy.signal import freqz
from filter_utilities_jos import check_roots_stability
from filter_plot_utilities_jos import zplane

def invfreqz(
    H: np.ndarray,
    n_zeros: int,
    n_poles: int,
    U: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    n_iter: int | None = 0,
    tolr: float | None = 1e-8        
) -> tuple[np.ndarray, np.ndarray]:

    if n_iter == 0:
        return fast_equation_error_filter_design(H, n_zeros, n_poles, U, weight, omega)
    else:
        return fast_steiglitz_mcbride_filter_design(
            H, U, n_zeros, n_poles,
            max_iterations=n_iter, tol_iteration_change=tolr,
            zero_clip=1e-7, stabilize=True)


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


# Return a complete spectrum given the non-negative-frequency portion,
# including dc and fs/2:
def append_flip_conjugate(X):
    """
    Append the flipped conjugate of the input array, excluding the
    first and last elements (normally dc and fs/2) in what is flipped and appended.

    Parameters:
    X (np.ndarray): Input array, typically a spectrum over [0,pi], inclusive.

    Returns:
    np.ndarray: Array with appended flipped conjugate interior, i.e., "[0,pi,pi-,0+]".

    """
    # return        np.concatenate([X, np.conj(X[-2:0:-1])])
    # equivalent to np.concatenate([X, np.conj(np.flip(X[1:-1]))])
    flip_conj_interior = np.conj(np.flip(X[1:-1])) # negative-frequency part
    return np.concatenate([X, flip_conj_interior]) # complete spectrum


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

    # breakpoint()

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
    # breakpoint()
    roots = np.roots(A)
    unstable_mask = np.abs(roots) > 1
    if not np.any(unstable_mask):
        return A, roots, True  # All roots are stable
    # Invert the unstable roots:
    roots[unstable_mask] = 1 / np.conj(roots[unstable_mask])
    A_stable = np.poly(roots) # reconstruct the polynomial coefficients
    A_stable = check_real(A_stable)
    return A_stable, roots, False  # Some roots were unstable and inverted

def fast_steiglitz_mcbride_filter_design(H, U, n_zeros, n_poles, max_iterations=0,
                                         tol_iteration_change=1e-7,
                                         zero_clip=1e-7, stabilize=True):
    """Frequency-domain Steiglitz-McBride algorithm.

    The Steiglitz-McBride algorithm converts an equation-error filter
    design to an output-error filter design.  To accomplish this, it
    iteratively calls fast_equation_error_filter_design applying the
    filter 1/a to both input and output on each iteration until either
    the maximum number of iterations is reached or the stopping
    tolerance in successive filter changes is achieved.

    Parameters:
    H (array): Desired frequency response, uniformly sampled,
               including dc and pi, with no negative frequencies
    U (array): Input frequency response (can be used for weighting)
    n_zeros (int): Number of zeros in the filter
    n_poles (int): Number of poles in the filter
    max_iterations (int): Max number of iterations of the Steiglitz-McBride algorithm
    tol_iteration_change (float): Tolerance on the norm of the coefficients changes
                                  at which to halt Steiglitz-McBride iterations.
    stabilize (bool): When true, reflect any unstable poles
                      inside the unit circle if they go unstable.

    Returns:
    b (array): Numerator coefficients of the designed filter
    a (array): Denominator coefficients of the designed filter

    For maximum efficiency, the number of frequency points (length of
    H and U) should be Nfft/2+1, where Nfft is a power of 2 (FFT size
    used herein).

    """

    current_b = np.zeros(n_zeros+1)
    current_a = np.hstack((1, np.zeros(n_poles)))
    iterations = 0

    N = len(H)
    w = np.linspace(0, np.pi, N)

    if U is None:  # Make it effectively all 1s (no frequency-weighting == "impulse")
        U = np.ones_like(H)

    # We are going to modify these arrays:
    H_local = H.copy()
    U_local = U.copy()

    breakpoint()

    while True:
        new_b, new_a = fast_equation_error_filter_design(
            H_local, n_zeros, n_poles, U=U_local, omega = w )
        if stabilize:
            new_a = invert_unstable_roots(new_a)
        freqz(new_b, new_a)
        zplane(new_b, new_a)
        norm_change = norm(new_a - current_a) + norm(new_b - current_b)
        print(f"norm_change in a at iteration {iterations}: {norm_change}")
        if norm_change < tol_iteration_change * norm(current_a):
            print(f"""
            Stopping tolerance {tol_iteration_change} reached
            after {iterations + 1} iterations.""")
            break
        if iterations > max_iterations:
            print(f"Reached maximum of {iterations} iterations.")
            break
        # Go around the horn again:
        current_a = new_a
        current_b = new_b
        iterations += 1
        wA, A = freqz(current_a, worN=N)
        Ai = clipped_real_array_inverse(A, zero_clip)
        H_local = H_local * Ai
        U_local = U_local * Ai

    return new_b, new_a

# Example usage:
if __name__ == "__main__":
    from scipy.signal import freqz

    import pdb
    pdb.set_trace()

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
                                                  max_iterations=30,
                                                  tol_iteration_change=1e-12)
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
