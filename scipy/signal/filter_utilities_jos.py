import numpy as np

def check_roots_stability(roots, tol=1e-7):
    """
    Check the stability of roots.
    
    Args:
    roots (array-like): Array of complex roots
    tol (float): Tolerance for marginal stability
    
    Returns:
    tuple: (number of unstable roots, number of marginally stable roots)

    Example usage:
    roots = np.array([0.5 + 0.5j, 1.0001, 0.9999, 1.5, 0.8])
    num_unstable, num_marginally_stable = check_roots_stability(roots, tol=1e-3)
    print(f"Number of unstable roots: {num_unstable}")
    print(f"Number of marginally stable roots: {num_marginally_stable}")

    """
    magnitudes = np.abs(roots)
    num_unstable = np.sum(magnitudes > 1.0 + tol)
    num_marginally_stable = np.sum((magnitudes >= 1.0 - tol) & \
                                   (magnitudes <= 1.0 + tol))
    return num_unstable, num_marginally_stable
