from typing import Tuple
import numpy as np

def interface_to_centers(
    x_interface: np.ndarray,
    D_interface: np.ndarray,
    method: str = "harmonic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Hummer-style interface diffusion values to bin-center values.

    Hummer's Bayesian estimator places D_{i+1/2} at internal bin *interfaces*.
    This routine reconstructs bin-center D_i by extrapolating one bin to each
    edge and then averaging (or taking harmonic means) of adjacent interface
    values.

    Parameters
    ----------
    x_interface : array (M,)
        CV positions of the M internal interfaces.
    D_interface : array (M,)
        D values at those interfaces.
    method : {'harmonic', 'avg'}
        ``'harmonic'`` (default) gives a better approximation for barriers
        (avoids over-estimating D at low-D regions); ``'avg'`` is simpler.

    Returns
    -------
    x_center : array (M+1,)
    D_center : array (M+1,)
    edges    : array (M+2,)  full edge positions used internally
    """
    xI = np.asarray(x_interface, dtype=float).ravel()
    DI = np.asarray(D_interface, dtype=float).ravel()
    if xI.size != DI.size:
        raise ValueError("x_interface and D_interface must have the same length.")
    if xI.size < 2:
        raise ValueError("Need at least 2 interface points.")

    dx_l = xI[1] - xI[0]
    dx_r = xI[-1] - xI[-2]
    edges = np.empty(xI.size + 2, dtype=float)
    edges[1:-1] = xI
    edges[0] = xI[0] - dx_l
    edges[-1] = xI[-1] + dx_r

    xC = 0.5 * (edges[:-1] + edges[1:])  # shape (M+1,)
    DC = np.empty(xC.size, dtype=float)
    DC[0] = DI[0]
    DC[-1] = DI[-1]

    if xC.size > 2:
        if method == "harmonic":
            with np.errstate(divide="ignore", invalid="ignore"):
                DC[1:-1] = 2.0 / (1.0 / DI[:-1] + 1.0 / DI[1:])
            bad = ~np.isfinite(DC[1:-1]) | (DC[1:-1] <= 0)
            DC[1:-1][bad] = 0.5 * (DI[:-1][bad] + DI[1:][bad])
        elif method == "avg":
            DC[1:-1] = 0.5 * (DI[:-1] + DI[1:])
        else:
            raise ValueError("method must be 'harmonic' or 'avg'.")

    return xC, DC, edges

def interpolate_D_to_grid(
    s_grid: np.ndarray,
    x_D: np.ndarray,
    D_vals: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a D(s) profile onto *s_grid*, clamping outside the range.

    Parameters
    ----------
    s_grid : array (N,)
        Target uniform grid.
    x_D : array (M,)
        CV positions of the diffusion profile.
    D_vals : array (M,)
        Diffusion values at those positions.
    method : {'linear', 'pchip'}
        Interpolation method.  ``'pchip'`` requires SciPy.

    Returns
    -------
    D_grid : array (N,)  non-negative finite diffusion values on s_grid.
    """
    s_grid = np.asarray(s_grid, dtype=float).ravel()
    x_D = np.asarray(x_D, dtype=float).ravel()
    D_vals = np.asarray(D_vals, dtype=float).ravel()

    good = D_vals[np.isfinite(D_vals) & (D_vals > 0)]
    if good.size == 0:
        raise ValueError("D profile has no positive finite values.")
    left = float(np.median(good[: min(3, good.size)]))
    right = float(np.median(good[max(0, good.size - 3):]))

    if method == "linear":
        Dg = np.interp(s_grid, x_D, D_vals, left=left, right=right)
    elif method == "pchip":
        from scipy.interpolate import PchipInterpolator  # type: ignore
        f = PchipInterpolator(x_D, D_vals, extrapolate=True)
        Dg = np.asarray(f(s_grid), dtype=float)
        Dg = np.where(s_grid < x_D[0], left, Dg)
        Dg = np.where(s_grid > x_D[-1], right, Dg)
    else:
        raise ValueError("method must be 'linear' or 'pchip'.")

    floor = float(good.min() * 1e-6)
    Dg = np.where(np.isfinite(Dg) & (Dg > 0), Dg, floor)
    return Dg
