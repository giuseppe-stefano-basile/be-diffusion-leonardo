"""stochkin.fpe
=============

Fokker–Planck equation (FPE) solvers and grid-based kinetic operators.

This module provides numerical tools for forward and backward
Smoluchowski / Fokker–Planck equations on free-energy surfaces:

**Forward FPE (probability evolution)**

- :func:`solve_fp_steady_state` – 2D time-stepping toward the steady-state
  density using FiPy (preferred) or an explicit NumPy fallback.
- :func:`solve_fp_1d_from_fes` – 1D time-stepping on a free-energy
  profile :math:`F(s)` with position-dependent diffusion :math:`D(s)`.

**Discrete Fokker–Planck generator**

- :func:`build_fp_generator_from_fes` – build a detailed-balance-preserving
  rate matrix :math:`L` on a 2D grid (SciPy sparse or dense).
- :func:`build_smolu_generator_1d` – build the corresponding nearest-neighbour
  Smoluchowski generator on a 1D free-energy grid.

**Backward BVP solvers (1D, pure NumPy)**

- :func:`solve_committor_1d_from_fes` – committor :math:`q(s)` from a
  tridiagonal backward equation.
- :func:`solve_exit_time_1d_from_fes` – mean exit time :math:`\\tau(s)`.
- :func:`compute_ctmc_generator_fpe_1d` – multi-basin CTMC generator
  from backward 1D solves.
- :func:`mfpt_1d_smolu_integral` – analytic Smoluchowski integral for
  :math:`\\tau_{i \\to j}`.

**Backward BVP solvers (2D, FiPy)**

- :func:`solve_committor_fipy` – 2D committor on a FiPy mesh.
- :func:`solve_exit_time_fipy` – 2D mean exit time.
- :func:`compute_ctmc_generator_fpe_fipy` – multi-basin CTMC generator
  from backward 2D FiPy solves.
"""

from __future__ import annotations

import warnings

import numpy as np

# -------------------------------------------------------------------------
# Optional progress bar
# -------------------------------------------------------------------------
try:  # pragma: no cover
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(it):
        return it

# -------------------------------------------------------------------------
# Optional matplotlib (only used for quick diagnostics / plots)
# -------------------------------------------------------------------------
try:  # pragma: no cover
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    plt = None
    _HAVE_MPL = False

# -------------------------------------------------------------------------
# Optional FiPy backend
# -------------------------------------------------------------------------
try:
    from fipy import (
        CellVariable,
        Grid1D,
        Grid2D,
        # PDE terms
        TransientTerm,
        DiffusionTerm,
        ExponentialConvectionTerm,
        ImplicitSourceTerm,
    )
    from fipy.tools import numerix

    try:
        from fipy.meshes.nonUniformGrid2D import NonUniformGrid2D
    except Exception:  # pragma: no cover
        try:
            from fipy.meshes import NonUniformGrid2D  # type: ignore
        except Exception:  # pragma: no cover
            NonUniformGrid2D = None

    _HAVE_FIPY = True
except Exception:  # pragma: no cover
    CellVariable = None  # type: ignore
    Grid1D = None  # type: ignore
    Grid2D = None  # type: ignore
    TransientTerm = None  # type: ignore
    DiffusionTerm = None  # type: ignore
    ExponentialConvectionTerm = None  # type: ignore
    ImplicitSourceTerm = None  # type: ignore
    numerix = None  # type: ignore
    NonUniformGrid2D = None  # type: ignore
    _HAVE_FIPY = False


def _default_fipy_solver(tolerance: float = 1e-10, iterations: int = 2000):
    """Return a reasonable default FiPy linear solver if available.

    FiPy solver class locations differ between installations; this helper tries
    a handful of common ones and falls back to FiPy's defaults if needed.
    """

    if not _HAVE_FIPY:
        return None

    for mod_name, cls_name in [
        ("fipy.solvers.scipy", "LinearGMRESSolver"),
        ("fipy.solvers.scipy", "LinearBicgstabSolver"),
        ("fipy.solvers.scipy", "LinearPCGSolver"),
        ("fipy.solvers.scipy", "LinearLUSolver"),
        ("fipy.solvers", "LinearGMRESSolver"),
        ("fipy.solvers", "LinearLUSolver"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            try:
                return cls(tolerance=float(tolerance), iterations=int(iterations))
            except TypeError:
                return cls(tolerance=float(tolerance))
        except Exception:
            continue

    return None


# -------------------------------------------------------------------------
# Optional SciPy sparse backend
# -------------------------------------------------------------------------
try:  # pragma: no cover
    import scipy.sparse as sp  # type: ignore

    _HAVE_SCIPY_SPARSE = True
except Exception:  # pragma: no cover
    sp = None
    _HAVE_SCIPY_SPARSE = False


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def _compute_potential_grid(potential, xs, ys):
    """Sample U(x,y) and ∇U(x,y) on a regular grid.

    Parameters
    ----------
    potential : callable
        potential([x, y]) -> (U, F) with F = -∇U.
    xs, ys : 1D arrays
        Grid coordinates.

    Returns
    -------
    U, Ux, Uy : (nx, ny)
        U(x,y), dU/dx, dU/dy on the grid.
    """

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    nx = xs.size
    ny = ys.size

    U = np.empty((nx, ny), dtype=float)
    Ux = np.empty_like(U)
    Uy = np.empty_like(U)

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            E, F = potential([xv, yv])
            U[i, j] = float(E)
            # F = -∇U ⇒ ∇U = -F
            Ux[i, j] = -float(F[0])
            Uy[i, j] = -float(F[1])

    return U, Ux, Uy


# -------------------------------------------------------------------------
#  Explicit NumPy fallback (conservative finite-volume scheme)
# -------------------------------------------------------------------------

def _solve_fp_steady_state_explicit(
    potential,
    xlim,
    ylim,
    nx,
    ny,
    D,
    beta,
    dt,
    n_steps,
    initial,
    normalize_each_step,
    plot_final,
):
    """Explicit finite-volume integration for the 2D forward FPE.

    This is a fallback when FiPy is not available.

    We integrate the divergence-form forward equation:

        ∂_t p = -∇·J
        J = -D (∇p + β p ∇U)

    with reflecting (no-flux) boundaries implemented by setting face fluxes
    to zero at the domain boundary.

    Notes
    -----
    This explicit scheme can require small `dt` for stability.
    """

    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])

    xs = np.linspace(x0, x1, int(nx))
    ys = np.linspace(y0, y1, int(ny))

    if xs.size < 2 or ys.size < 2:
        raise ValueError("nx and ny must be >= 2.")

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    U, Ux, Uy = _compute_potential_grid(potential, xs, ys)

    # Initial condition
    if initial == "uniform":
        p = np.ones((xs.size, ys.size), dtype=float)
    elif callable(initial):
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        coords = np.vstack((X.ravel(), Y.ravel()))
        p0 = np.asarray(initial(coords), dtype=float).ravel()
        if p0.size != xs.size * ys.size:
            raise ValueError(
                f"Initial condition callable must return {xs.size*ys.size} values, got {p0.size}."
            )
        p = np.maximum(p0.reshape((xs.size, ys.size)), 0.0)
    else:
        raise ValueError("initial must be 'uniform' or a callable(coords)->array")

    Dbeta = float(D) * float(beta)

    def _normalize():
        total = float(p.sum() * dx * dy)
        if total > 0.0:
            p[:] = p / total

    _normalize()

    # Face flux arrays
    Jx = np.zeros((xs.size + 1, ys.size), dtype=float)
    Jy = np.zeros((xs.size, ys.size + 1), dtype=float)

    for _ in range(int(n_steps)):
        # X-fluxes on vertical faces (i+1/2, j)
        pL = p[:-1, :]
        pR = p[1:, :]
        p_face = 0.5 * (pL + pR)

        Ux_face = 0.5 * (Ux[:-1, :] + Ux[1:, :])
        dpdx = (pR - pL) / dx
        Jx[1:-1, :] = -float(D) * (dpdx + Dbeta * p_face * Ux_face)

        # Reflecting boundary faces
        Jx[0, :] = 0.0
        Jx[-1, :] = 0.0

        # Y-fluxes on horizontal faces (i, j+1/2)
        pB = p[:, :-1]
        pT = p[:, 1:]
        p_face = 0.5 * (pB + pT)

        Uy_face = 0.5 * (Uy[:, :-1] + Uy[:, 1:])
        dpdy = (pT - pB) / dy
        Jy[:, 1:-1] = -float(D) * (dpdy + Dbeta * p_face * Uy_face)

        Jy[:, 0] = 0.0
        Jy[:, -1] = 0.0

        # Divergence (cell-centered)
        dJx_dx = (Jx[1:, :] - Jx[:-1, :]) / dx
        dJy_dy = (Jy[:, 1:] - Jy[:, :-1]) / dy
        dpdt = -(dJx_dx + dJy_dy)

        p[:] = np.maximum(p + float(dt) * dpdt, 0.0)

        if normalize_each_step:
            _normalize()

    _normalize()

    if plot_final and _HAVE_MPL:
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        fig, ax = plt.subplots()
        c = ax.pcolormesh(X, Y, p, shading="auto")
        fig.colorbar(c, ax=ax, label="p(x,y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Steady-state probability density p(x,y) (explicit solver)")
        ax.set_aspect("equal")
        plt.show()

    return {
        "p_grid": p,
        "xs": xs,
        "ys": ys,
        "U_grid": U,
        "Ux": Ux,
        "Uy": Uy,
    }


# -------------------------------------------------------------------------
#  FiPy-based implicit solver (preferred)
# -------------------------------------------------------------------------

def solve_fp_steady_state(
    potential,
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
    nx=100,
    ny=100,
    D=1.0,
    beta=1.0,
    dt=1e-3,
    n_steps=500,
    initial="uniform",
    normalize_each_step=True,
    viewer=False,
    plot_final=True,
):
    """Solve the 2D forward Fokker–Planck equation toward steady state.

    Evolves the probability density :math:`p(x,y,t)` of the overdamped
    Langevin equation

    .. math::

       dX_t = -D\\beta\\,\\nabla U(X_t)\\,dt + \\sqrt{2D}\\,dW_t

    via the forward FPE in divergence form:

    .. math::

       \\partial_t p = \\nabla\\cdot(D\\nabla p + D\\beta\\, p\\,\\nabla U)

    Uses FiPy (implicit, preferred) when available; otherwise falls back
    to an explicit NumPy FTCS scheme.

    Parameters
    ----------
    potential : callable
        ``potential([x, y]) -> (U, F)`` with ``F = -\\nabla U``.
    xlim, ylim : (float, float)
        Domain bounds.
    nx, ny : int
        Grid points per axis.
    D : float
        Diffusion coefficient (constant).
    beta : float
        Inverse temperature.
    dt : float
        Time step.
    n_steps : int
        Number of integration steps.
    initial : {'uniform'} or callable
        Initial distribution.
    normalize_each_step : bool
        Re-normalise :math:`p` at every step.
    viewer : bool
        Use FiPy’s live viewer (FiPy only).
    plot_final : bool
        Plot the final density.

    Returns
    -------
    dict
        Keys: ``'p_grid'``, ``'xs'``, ``'ys'``, ``'U_grid'``,
        ``'Ux'``, ``'Uy'``.
    """

    if not _HAVE_FIPY:
        return _solve_fp_steady_state_explicit(
            potential=potential,
            xlim=xlim,
            ylim=ylim,
            nx=nx,
            ny=ny,
            D=D,
            beta=beta,
            dt=dt,
            n_steps=n_steps,
            initial=initial,
            normalize_each_step=normalize_each_step,
            plot_final=plot_final,
        )

    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])

    Lx = x1 - x0
    Ly = y1 - y0

    nx = int(nx)
    ny = int(ny)
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")

    dx = Lx / nx
    dy = Ly / ny

    # Base mesh [0,Lx]×[0,Ly] then translate.
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    mesh = mesh + ((x0,), (y0,))

    # Sample potential and gradient on FiPy cell centers
    cx = np.asarray(mesh.cellCenters[0], dtype=float)
    cy = np.asarray(mesh.cellCenters[1], dtype=float)
    n_cells = cx.size

    U_vals = np.empty(n_cells, dtype=float)
    Ux_vals = np.empty_like(U_vals)
    Uy_vals = np.empty_like(U_vals)

    for k in range(n_cells):
        E, F = potential([cx[k], cy[k]])
        U_vals[k] = float(E)
        Ux_vals[k] = -float(F[0])
        Uy_vals[k] = -float(F[1])

    Dbeta = float(D) * float(beta)
    convCoeff = CellVariable(mesh=mesh, name="Dbeta_gradU", rank=1)
    convCoeff.value = np.vstack((Dbeta * Ux_vals, Dbeta * Uy_vals))

    # Initial condition
    if initial == "uniform":
        p0 = np.ones(n_cells, dtype=float)
    elif callable(initial):
        coords = np.vstack((cx, cy))
        p0 = np.asarray(initial(coords), dtype=float).ravel()
        if p0.size != n_cells:
            raise ValueError(
                f"Initial callable must return array of length {n_cells}, got {p0.size}."
            )
        p0 = np.maximum(p0, 0.0)
    else:
        raise ValueError("initial must be 'uniform' or a callable(coords)->array")

    p = CellVariable(mesh=mesh, name="p", value=p0)

    volumes = np.asarray(mesh.cellVolumes, dtype=float)

    def _normalize():
        total = float((p.value * volumes).sum())
        if total > 0.0:
            p.setValue(p.value / total)

    _normalize()

    view = None
    if viewer:
        try:  # pragma: no cover
            from fipy import Viewer

            view = Viewer(vars=p)
        except Exception:
            view = None

    # FiPy equation: ∂_t p = ∇·(D ∇p) + ∇·(convCoeff * p)
    eq = (
        TransientTerm(var=p)
        == DiffusionTerm(coeff=float(D), var=p)
        + ExponentialConvectionTerm(coeff=convCoeff, var=p)
    )

    for _ in _tqdm(range(int(n_steps))):
        eq.solve(var=p, dt=float(dt))
        p.setValue(numerix.where(p.value < 0.0, 0.0, p.value))

        if normalize_each_step:
            _normalize()

        if view is not None:
            view.plot()

    _normalize()

    # Map FiPy cells back to grid arrays
    xs = x0 + dx * (np.arange(nx) + 0.5)
    ys = y0 + dy * (np.arange(ny) + 0.5)

    p_grid = np.zeros((nx, ny), dtype=float)
    U_grid = np.zeros((nx, ny), dtype=float)
    Ux_grid = np.zeros((nx, ny), dtype=float)
    Uy_grid = np.zeros((nx, ny), dtype=float)

    i_idx = np.floor((cx - x0) / dx).astype(int)
    j_idx = np.floor((cy - y0) / dy).astype(int)

    p_vals = np.asarray(p.value, dtype=float)

    p_grid[i_idx, j_idx] = p_vals
    U_grid[i_idx, j_idx] = U_vals
    Ux_grid[i_idx, j_idx] = Ux_vals
    Uy_grid[i_idx, j_idx] = Uy_vals

    if plot_final and _HAVE_MPL:
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        fig, ax = plt.subplots()
        c = ax.pcolormesh(X, Y, p_grid, shading="auto")
        fig.colorbar(c, ax=ax, label="p(x,y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Steady-state probability density p(x,y) (FiPy solver)")
        ax.set_aspect("equal")
        plt.show()

    return {
        "p_grid": p_grid,
        "xs": xs,
        "ys": ys,
        "U_grid": U_grid,
        "Ux": Ux_grid,
        "Uy": Uy_grid,
    }


# -------------------------------------------------------------------------
# Discrete FP generator on a 2D FES grid (for kinetic analysis)
# -------------------------------------------------------------------------

def build_fp_generator_from_fes(xs, ys, U_grid, D, beta, sparse=True):
    r"""Build a detailed-balance-preserving discrete FP generator on a regular grid.

    Parameters
    ----------
    xs, ys : 1D arrays
        Grid coordinates (monotonic, uniform spacing expected).
    U_grid : (nx, ny) array
        Potential / free energy on the grid.
    D : float or (nx, ny) array
        Diffusion coefficient (scalar or field).
    beta : float
        Inverse temperature.
    sparse : bool
        If True, return a SciPy CSR matrix (requires SciPy). If False, return dense.

    Returns
    -------
    L : (N, N)
        Generator with off-diagonals >=0 and row sums 0. Indexing: k=i*ny + j.

    Notes
    -----
    A common symmetric ("square-root") discretization is used:

    .. math::

        k_{i \to j} = \frac{D_{\text{face}}}{\Delta^2}
        \exp\!\bigl[-\tfrac{\beta}{2}(U_j - U_i)\bigr]

    so that for constant *D* the stationary distribution satisfies
    :math:`\pi_i \propto \exp(-\beta U_i)` up to discretization error.
    """

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    U = np.asarray(U_grid, dtype=float)

    if U.ndim != 2:
        raise ValueError("U_grid must be 2D with shape (nx, ny).")

    nx, ny = U.shape

    # Ensure coordinate vectors match the grid shape
    if xs.size != nx or ys.size != ny:
        raise ValueError(
            f"xs/ys lengths ({xs.size},{ys.size}) do not match U_grid shape ({nx},{ny})."
        )

    if not np.all(np.isfinite(U)):
        raise ValueError(
            "U_grid contains non-finite values; mask/crop invalid regions before building L."
        )

    if np.isscalar(D):
        D_grid = np.full_like(U, float(D), dtype=float)
    else:
        D_arr = np.asarray(D, dtype=float)
        if D_arr.shape != U.shape:
            raise ValueError(f"D has shape {D_arr.shape}, expected scalar or {U.shape}.")
        if not np.all(np.isfinite(D_arr)):
            raise ValueError("D contains non-finite values.")
        if np.any(D_arr <= 0.0):
            raise ValueError("D must be strictly positive everywhere.")
        D_grid = D_arr

    if xs.size < 2 or ys.size < 2:
        raise ValueError("xs and ys must have length >= 2")

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Grid coordinates xs, ys must be strictly increasing")

    # Validate (near-)uniform spacing. The symmetric detailed-balance discretization
    # used below assumes constant dx, dy.
    dxs = np.diff(xs)
    dys = np.diff(ys)
    if not (np.allclose(dxs, dx, rtol=1e-6, atol=1e-12) and np.allclose(dys, dy, rtol=1e-6, atol=1e-12)):
        raise ValueError(
            "Non-uniform grid spacing detected in xs/ys. "
            "build_fp_generator_from_fes currently requires a uniform grid. "
            "Resample/interpolate to a regular grid or implement a non-uniform discretization."
        )

    N = nx * ny

    if sparse:
        if not _HAVE_SCIPY_SPARSE:
            raise ImportError(
                "build_fp_generator_from_fes(sparse=True) requires scipy.sparse. "
                "Install SciPy or call with sparse=False (small grids only)."
            )
        L = sp.lil_matrix((N, N), dtype=float)
    else:
        L = np.zeros((N, N), dtype=float)

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            row = idx(i, j)
            Ui = U[i, j]
            Di = D_grid[i, j]
            rate_sum = 0.0

            # +x
            if i + 1 < nx:
                Uj = U[i + 1, j]
                Dj = D_grid[i + 1, j]
                D_face = 0.5 * (Di + Dj)
                k = D_face / (dx * dx) * np.exp(-0.5 * float(beta) * (Uj - Ui))
                L[row, idx(i + 1, j)] = k
                rate_sum += k

            # -x
            if i - 1 >= 0:
                Uj = U[i - 1, j]
                Dj = D_grid[i - 1, j]
                D_face = 0.5 * (Di + Dj)
                k = D_face / (dx * dx) * np.exp(-0.5 * float(beta) * (Uj - Ui))
                L[row, idx(i - 1, j)] = k
                rate_sum += k

            # +y
            if j + 1 < ny:
                Uj = U[i, j + 1]
                Dj = D_grid[i, j + 1]
                D_face = 0.5 * (Di + Dj)
                k = D_face / (dy * dy) * np.exp(-0.5 * float(beta) * (Uj - Ui))
                L[row, idx(i, j + 1)] = k
                rate_sum += k

            # -y
            if j - 1 >= 0:
                Uj = U[i, j - 1]
                Dj = D_grid[i, j - 1]
                D_face = 0.5 * (Di + Dj)
                k = D_face / (dy * dy) * np.exp(-0.5 * float(beta) * (Uj - Ui))
                L[row, idx(i, j - 1)] = k
                rate_sum += k

            L[row, row] = -rate_sum

    return L.tocsr() if sparse else L


# -------------------------------------------------------------------------
# 1D forward FPE on a free energy profile (FiPy)
# -------------------------------------------------------------------------

def solve_fp_1d_from_fes(
    s,
    F,
    D,
    beta=1.0,
    dt=1e-3,
    n_steps=1000,
    initial="boltzmann",
    normalize_each_step=True,
    viewer=False,
):
    """1D forward Fokker–Planck solver on a free-energy profile.

    Solves

    .. math::

       \\partial_t p(s,t) = \\partial_s\\bigl[D(s)\\bigl(
       \\partial_s p + \\beta\\, p\\, F'(s)\\bigr)\\bigr]

    with reflecting (no-flux) boundaries via FiPy.

    Parameters
    ----------
    s : array_like
        Uniform 1D grid.
    F : array_like
        Free-energy values on *s*.
    D : array_like
        Diffusion coefficient on *s*.
    beta : float
        Inverse temperature.
    dt : float
        Time step.
    n_steps : int
        Number of integration steps.
    initial : {'uniform', 'boltzmann'} or array_like
        Initial distribution.
    normalize_each_step : bool
        Re-normalise at every step.
    viewer : bool
        Use FiPy’s live viewer.

    Returns
    -------
    dict
        Keys: ``'s'``, ``'p'``, ``'F'``, ``'D'``, ``'dF_ds'``.

    Raises
    ------
    ImportError
        If FiPy is not installed.
    """

    s = np.asarray(s, dtype=float)
    F = np.asarray(F, dtype=float)
    D = np.asarray(D, dtype=float)

    if s.ndim != 1:
        raise ValueError("s must be a 1D grid")
    if not (F.shape == s.shape and D.shape == s.shape):
        raise ValueError("F and D must have the same shape as s")

    if not _HAVE_FIPY:
        raise ImportError("FiPy is required for solve_fp_1d_from_fes()")

    ns = s.size
    if ns < 3:
        raise ValueError("Need at least 3 grid points")

    ds_arr = np.diff(s)
    ds = float(ds_arr.mean())
    if np.max(np.abs(ds_arr - ds)) > 1e-6 * abs(ds):
        raise ValueError("solve_fp_1d_from_fes currently assumes a uniform grid in s")

    x0 = float(s[0] - 0.5 * ds)
    mesh = Grid1D(dx=ds, nx=ns) + ((x0,),)

    cell_s = np.asarray(mesh.cellCenters[0], dtype=float)

    F_vals = np.interp(cell_s, s, F)
    D_vals = np.interp(cell_s, s, D)

    dF_ds = np.gradient(F_vals, cell_s)

    D_var = CellVariable(mesh=mesh, name="D(s)", value=D_vals)

    # ExponentialConvectionTerm expects a *vector* coefficient.
    # In 1D, that is shape (1, nCells).
    convCoeff = CellVariable(mesh=mesh, name="Dbeta_dFds", rank=1)
    convCoeff.value = np.array([D_vals * float(beta) * dF_ds], dtype=float)

    # Initial condition
    if isinstance(initial, str):
        if initial.lower() == "uniform":
            p0 = np.ones_like(cell_s)
        elif initial.lower() == "boltzmann":
            p0 = np.exp(-float(beta) * F_vals)
        else:
            raise ValueError("initial must be 'uniform', 'boltzmann', or an array")
    else:
        p0 = np.asarray(initial, dtype=float)
        if p0.shape != cell_s.shape:
            raise ValueError("Initial array must have the same shape as s")
        p0 = np.maximum(p0, 0.0)

    p = CellVariable(mesh=mesh, name="p(s)", value=p0)

    volumes = np.asarray(mesh.cellVolumes, dtype=float)

    def _normalize():
        total = float((p.value * volumes).sum())
        if total > 0.0:
            p.setValue(p.value / total)

    _normalize()

    view = None
    if viewer:
        try:  # pragma: no cover
            from fipy import Viewer

            view = Viewer(vars=p)
        except Exception:
            view = None

    eq = (
        TransientTerm(var=p)
        == DiffusionTerm(coeff=D_var, var=p)
        + ExponentialConvectionTerm(coeff=convCoeff, var=p)
    )

    for _ in _tqdm(range(int(n_steps))):
        eq.solve(var=p, dt=float(dt))
        p.setValue(numerix.where(p.value < 0.0, 0.0, p.value))
        if normalize_each_step:
            _normalize()
        if view is not None:
            view.plot()

    _normalize()

    return {
        "s": cell_s.copy(),
        "p": np.asarray(p.value, dtype=float),
        "F": F_vals,
        "D": D_vals,
        "dF_ds": dF_ds,
    }


def mfpt_1d_smolu_integral(s, F, D, beta, i_index, j_index):
    """Analytic Smoluchowski integral for the 1D MFPT :math:`\\tau_{i\\to j}`.

    Uses the classical formula

    .. math::

       \\tau_{i\\to j}
         = \\int_{s_i}^{s_j}
             \\frac{e^{\\beta F(z)}}{D(z)}
             \\int_{s_{\\min}}^{z} e^{-\\beta F(y)}\\,dy\\;dz

    evaluated via the trapezoidal rule.

    Parameters
    ----------
    s : array_like
        1D grid (must be sorted).
    F : array_like
        Free-energy values on *s*.
    D : array_like
        Diffusion coefficient on *s*.
    beta : float
        Inverse temperature.
    i_index, j_index : int
        Grid indices of source and target.

    Returns
    -------
    float
        Mean first-passage time :math:`\\tau_{i\\to j}`.
    """

    s = np.asarray(s, dtype=float)
    F = np.asarray(F, dtype=float)
    D = np.asarray(D, dtype=float)

    if not (s.ndim == F.ndim == D.ndim == 1):
        raise ValueError("s, F and D must be 1D arrays")
    if not (s.shape == F.shape == D.shape):
        raise ValueError("s, F and D must have the same shape")

    n = s.size
    if i_index == j_index:
        return 0.0
    if not (0 <= i_index < n and 0 <= j_index < n):
        raise IndexError("i_index and j_index must be valid indices")

    # If target is left of start, reverse arrays to reuse same formula.
    if j_index < i_index:
        s = s[::-1]
        F = F[::-1]
        D = D[::-1]
        i_index = n - 1 - i_index
        j_index = n - 1 - j_index

    f = np.exp(-float(beta) * F)

    # A(z) = ∫_{s_min}^{z} e^{-βF}
    A = np.zeros_like(s)
    for k in range(1, n):
        ds = s[k] - s[k - 1]
        A[k] = A[k - 1] + 0.5 * (f[k - 1] + f[k]) * ds

    g = np.exp(float(beta) * F) * A / D

    tau = 0.0
    for k in range(i_index + 1, j_index + 1):
        ds = s[k] - s[k - 1]
        tau += 0.5 * (g[k - 1] + g[k]) * ds

    return float(tau)


# =============================================================================
# 1D backward Smoluchowski solves (NumPy tridiagonal; no FiPy required)
#
# We use the same divergence-form coefficient as the 2D FiPy backward solver:
#
#   d/ds ( A(s) d u/ds ) = b(s)
#
# with
#   w(s) = exp(-beta * (F(s) - Fmin))
#   A(s) = D(s) * w(s)
#
# This corresponds to the backward generator
#   L u = (1/w) d/ds ( D w d u/ds )
# and yields the standard committor and exit-time BVPs:
#   d/ds (A q') = 0, with q=1 on B and q=0 on A
#   d/ds (A tau') = -w, with tau=0 on absorbing set
#
# We discretize on a uniform 1D grid using conservative face fluxes.
# =============================================================================


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """Compute exp(x) with clipping to avoid overflow/underflow."""

    x = np.asarray(x, dtype=float)
    return np.exp(np.clip(x, -700.0, 700.0))


def _require_uniform_grid_1d(s: np.ndarray, rtol: float = 1e-10, atol: float = 1e-12) -> float:
    """Return uniform spacing ds, or raise if s is not (approximately) uniform."""

    s = np.asarray(s, dtype=float).ravel()
    if s.size < 3:
        raise ValueError("Need at least 3 grid points")
    ds = np.diff(s)
    if not np.all(ds > 0):
        raise ValueError("Grid s must be strictly increasing")
    ds0 = float(ds.mean())
    if not np.allclose(ds, ds0, rtol=rtol, atol=atol):
        raise ValueError(
            "This solver requires an (approximately) uniform grid in s. "
            "Resample/interpolate your FES to a uniform grid first."
        )
    return ds0


def _as_1d_array(x, n: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 1:
        return np.full(n, float(x[0]), dtype=float)
    if x.size != n:
        raise ValueError(f"{name} must be scalar or have length {n}")
    return x


def _as_1d_face_array(x, n: int, name: str) -> np.ndarray:
    """Return face-centered values of length n-1 from scalar/cell/face input."""

    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 1:
        value = float(arr[0])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be strictly positive")
        return np.full(n - 1, value, dtype=float)
    if arr.size == n:
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} must be strictly positive and finite")
        return 0.5 * (arr[:-1] + arr[1:])
    if arr.size == n - 1:
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"{name} must be strictly positive and finite")
        return arr.astype(float, copy=True)
    raise ValueError(f"{name} must be scalar, length {n}, or face-centered length {n - 1}")


def build_smolu_generator_1d(
    s,
    F,
    D,
    beta=1.0,
    *,
    boundary="reflecting",
    convention="row",
    sparse=False,
    validate=True,
):
    r"""Build a detailed-balance-preserving 1D Smoluchowski generator.

    The row-convention generator has entries ``K[i, j] = rate i -> j`` and
    rows summing to zero.  Rates use the symmetric discretization

    .. math::

        k_{i \to i+1}
        =
        \frac{D_{i+1/2}}{\Delta s^2}
        \exp[-\tfrac{\beta}{2}(F_{i+1}-F_i)].

    Parameters
    ----------
    s : array_like, shape (n,)
        Uniform, strictly increasing CV grid.
    F : array_like, shape (n,)
        Free-energy profile on ``s``.
    D : float or array_like
        Diffusion coefficient.  Scalars are constant; length ``n`` arrays are
        cell-centered; length ``n-1`` arrays are face-centered.
    beta : float
        Inverse energy, compatible with ``F``.
    boundary : {"reflecting", "reflect"}
        Reflecting/no-flux boundaries.  Other boundaries are not implemented
        in this first version.
    convention : {"row", "column"}
        Return row-sum-zero generator by default; ``"column"`` returns the
        transpose.
    sparse : bool
        Return SciPy CSR when True.
    validate : bool
        Check finite inputs and positive diffusion.
    """

    s = np.asarray(s, dtype=float).ravel()
    F = np.asarray(F, dtype=float).ravel()
    if s.size != F.size:
        raise ValueError("s and F must have the same length")
    if s.size < 3:
        raise ValueError("Need at least 3 grid points")
    ds = _require_uniform_grid_1d(s)

    if str(boundary).lower() not in {"reflecting", "reflect", "noflux", "no-flux"}:
        raise ValueError("build_smolu_generator_1d currently supports reflecting boundaries only")
    conv = str(convention).lower()
    if conv not in {"row", "column"}:
        raise ValueError("convention must be 'row' or 'column'")
    if not np.isfinite(float(beta)):
        raise ValueError("beta must be finite")
    if validate:
        if not np.all(np.isfinite(F)):
            raise ValueError("F contains non-finite values")

    D_face = _as_1d_face_array(D, s.size, "D")
    K = np.zeros((s.size, s.size), dtype=float)
    inv_ds2 = 1.0 / (float(ds) * float(ds))
    beta_val = float(beta)

    for i in range(s.size - 1):
        dF = float(F[i + 1] - F[i])
        rate_forward = float(D_face[i]) * inv_ds2 * np.exp(-0.5 * beta_val * dF)
        rate_backward = float(D_face[i]) * inv_ds2 * np.exp(+0.5 * beta_val * dF)
        K[i, i + 1] = rate_forward
        K[i + 1, i] = rate_backward

    for i in range(s.size):
        K[i, i] = -float(np.sum(K[i, :]))

    if conv == "column":
        K = K.T

    if sparse:
        if not _HAVE_SCIPY_SPARSE:
            raise ImportError(
                "build_smolu_generator_1d(sparse=True) requires scipy.sparse. "
                "Install SciPy or call with sparse=False."
            )
        return sp.csr_matrix(K)
    return K


def _solve_generator_dirichlet(K: np.ndarray, fixed_mask: np.ndarray, fixed_values: np.ndarray, rhs=None):
    """Solve ``K u = rhs`` on free states with fixed Dirichlet values."""

    K = np.asarray(K, dtype=float)
    fixed_mask = np.asarray(fixed_mask, dtype=bool).ravel()
    fixed_values = np.asarray(fixed_values, dtype=float).ravel()
    n = K.shape[0]
    if fixed_mask.size != n or fixed_values.size != n:
        raise ValueError("Dirichlet masks/values must match K shape")

    free_mask = ~fixed_mask
    free_idx = np.flatnonzero(free_mask)
    fixed_idx = np.flatnonzero(fixed_mask)
    u = fixed_values.copy()
    if free_idx.size == 0:
        return u

    A = K[np.ix_(free_idx, free_idx)]
    if rhs is None:
        b = np.zeros(free_idx.size, dtype=float)
    else:
        rhs_arr = np.asarray(rhs, dtype=float).ravel()
        if rhs_arr.size != n:
            raise ValueError("rhs must match K shape")
        b = rhs_arr[free_idx].copy()
    if fixed_idx.size > 0:
        b = b - K[np.ix_(free_idx, fixed_idx)] @ fixed_values[fixed_idx]
    try:
        u[free_idx] = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        u[free_idx] = np.linalg.lstsq(A, b, rcond=None)[0]
    return u


def compute_ctmc_generator_from_grid_generator_1d(
    s,
    F,
    K_grid,
    basin_network,
    *,
    beta=None,
    init_weight="boltzmann",
    convention="row",
    verbose=True,
):
    """Coarse-grain a 1D grid-state generator into basin CTMC rates.

    This is the finite-state analogue of :func:`compute_ctmc_generator_fpe_1d`.
    It accepts any row-sum-zero grid generator on the same grid as ``s`` and
    the basin labels, including a memory-corrected effective generator.

    Parameters
    ----------
    s, F : array_like, shape (n,)
        CV grid and free-energy profile used for basin averaging.
    K_grid : array_like, shape (n, n)
        Grid-state generator.  Row convention is used by default.
    basin_network : BasinNetwork1D
        Basin labels on the same grid.
    beta : float, optional
        Required for ``init_weight='boltzmann'``.
    init_weight : {'boltzmann', 'uniform'}
        Weight for basin averages of exit times and committors.
    convention : {'row', 'column'}
        Matrix orientation of ``K_grid``.

    Returns
    -------
    dict
        Keys include ``K``, ``exit_mean``, ``k_out``, ``p_branch``,
        ``basin_ids``, ``labels_full``, ``method`` and ``diagnostics``.
    """

    s = np.asarray(s, dtype=float).ravel()
    F = np.asarray(F, dtype=float).ravel()
    if s.size != F.size:
        raise ValueError("s and F must have the same length")
    n_grid = int(s.size)

    K_arr = K_grid.toarray() if hasattr(K_grid, "toarray") else K_grid
    K_arr = np.asarray(K_arr, dtype=float)
    if K_arr.shape != (n_grid, n_grid):
        raise ValueError(f"K_grid has shape {K_arr.shape}, expected {(n_grid, n_grid)}")
    if not np.all(np.isfinite(K_arr)):
        raise ValueError("K_grid contains non-finite values")

    conv = str(convention).lower()
    if conv not in {"row", "column"}:
        raise ValueError("convention must be 'row' or 'column'")
    K_row = K_arr if conv == "row" else K_arr.T

    labels_src = getattr(basin_network, "core_labels", None)
    if labels_src is None:
        labels_src = basin_network.labels
    labels = np.asarray(labels_src, dtype=int).ravel()
    labels_full = np.asarray(basin_network.labels, dtype=int).ravel()
    if labels.size != n_grid or labels_full.size != n_grid:
        raise ValueError("basin_network labels must have the same length as s")

    basin_ids = np.unique(labels[labels >= 0])
    basin_ids = np.sort(basin_ids)
    n_basins = int(basin_ids.size)
    if n_basins < 1:
        raise ValueError("No basins found (labels>=0) — cannot build CTMC")

    if str(init_weight).lower() == "uniform":
        weight = np.ones_like(F, dtype=float)
    elif str(init_weight).lower() == "boltzmann":
        if beta is None:
            raise ValueError("beta is required when init_weight='boltzmann'")
        F0 = float(np.nanmin(F[np.isfinite(F)]))
        weight = _safe_exp(-float(beta) * (F - F0))
    else:
        raise ValueError("init_weight must be 'boltzmann' or 'uniform'")

    in_any_basin = labels >= 0
    basin_masks = [(labels == bid) for bid in basin_ids]
    exit_mean = np.full(n_basins, np.nan, dtype=float)
    k_out = np.full(n_basins, np.nan, dtype=float)
    p_branch = np.full((n_basins, n_basins), np.nan, dtype=float)
    K_basin = np.zeros((n_basins, n_basins), dtype=float)

    if verbose:
        print(f"[grid CTMC] n_basins={n_basins}, init_weight={init_weight}")

    for i in range(n_basins):
        mask_i = basin_masks[i]
        if not np.any(mask_i):
            continue
        absorb = in_any_basin & (~mask_i)
        if not np.any(absorb):
            continue
        fixed_values = np.zeros(n_grid, dtype=float)
        tau = _solve_generator_dirichlet(
            K_row,
            fixed_mask=absorb,
            fixed_values=fixed_values,
            rhs=-np.ones(n_grid, dtype=float),
        )
        tau_mean = _weighted_average_1d(tau, weight, mask_i)
        exit_mean[i] = float(tau_mean) if np.isfinite(tau_mean) else np.nan
        if np.isfinite(exit_mean[i]) and abs(exit_mean[i]) > 0.0:
            k_out[i] = 1.0 / float(exit_mean[i])
        if verbose:
            print(f"[grid CTMC] basin {basin_ids[i]}: <tau_exit>={exit_mean[i]:.6g}, k_out={k_out[i]:.6g}")

    if n_basins == 2:
        for i in range(n_basins):
            if np.isfinite(k_out[i]):
                j = 1 - i
                p_branch[i, j] = 1.0
                K_basin[i, j] = float(k_out[i])
                K_basin[i, i] = -float(k_out[i])
    else:
        for i in range(n_basins):
            if not np.isfinite(k_out[i]):
                continue
            mask_i = basin_masks[i]
            for j in range(n_basins):
                if i == j:
                    continue
                mask_j = basin_masks[j]
                if not np.any(mask_j):
                    continue
                fixed_mask = in_any_basin & (~mask_i)
                fixed_values = np.zeros(n_grid, dtype=float)
                fixed_values[mask_j] = 1.0
                q = _solve_generator_dirichlet(
                    K_row,
                    fixed_mask=fixed_mask,
                    fixed_values=fixed_values,
                    rhs=np.zeros(n_grid, dtype=float),
                )
                p_branch[i, j] = _weighted_average_1d(q, weight, mask_i)

            row = p_branch[i, :].copy()
            row[i] = np.nan
            srow = np.nansum(row)
            if np.isfinite(srow) and srow != 0.0:
                for j in range(n_basins):
                    if i != j and np.isfinite(p_branch[i, j]):
                        p_branch[i, j] /= float(srow)
                        K_basin[i, j] = float(k_out[i] * p_branch[i, j])
            K_basin[i, i] = -float(np.sum(K_basin[i, :]))

    row_sums = K_row.sum(axis=1)
    offdiag = K_row.copy()
    np.fill_diagonal(offdiag, 0.0)
    basin_offdiag = K_basin.copy()
    np.fill_diagonal(basin_offdiag, 0.0)
    diagnostics = {
        "grid_max_abs_row_sum": float(np.max(np.abs(row_sums))) if row_sums.size else 0.0,
        "grid_min_offdiagonal": float(np.min(offdiag)) if offdiag.size else 0.0,
        "grid_negative_offdiagonal_count": int(np.sum(offdiag < -1e-12)),
        "basin_min_offdiagonal": float(np.min(basin_offdiag)) if basin_offdiag.size else 0.0,
        "basin_negative_offdiagonal_count": int(np.sum(basin_offdiag < -1e-12)),
        "branch_min": float(np.nanmin(p_branch)) if np.any(np.isfinite(p_branch)) else np.nan,
        "branch_max": float(np.nanmax(p_branch)) if np.any(np.isfinite(p_branch)) else np.nan,
    }
    if diagnostics["grid_negative_offdiagonal_count"] > 0 or diagnostics["basin_negative_offdiagonal_count"] > 0:
        warnings.warn(
            "Coarse-grained generator has negative off-diagonal entries; "
            "inspect diagnostics before interpreting rates.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "K": K_basin,
        "exit_mean": exit_mean,
        "k_out": k_out,
        "p_branch": p_branch,
        "basin_ids": basin_ids,
        "labels_full": labels_full,
        "method": "grid_generator_1d",
        "diagnostics": diagnostics,
    }


def _build_tridiag_div_A_grad_1d(A_cell: np.ndarray, ds: float):
    """Build tridiagonal for u -> -d/ds(A du/ds) with Neumann at domain ends.

    Notes
    -----
    We return the matrix for the *negative* divergence operator so that the
    resulting discrete operator matches the sign convention of a Markov
    generator (diagonal negative, off-diagonal positive) when used as
    L = (1/w) * (-div(A grad ·)).
    """

    A_cell = np.asarray(A_cell, dtype=float).ravel()
    n = int(A_cell.size)
    if n < 3:
        raise ValueError("Need at least 3 cells")

    # arithmetic face values (FiPy's arithmeticFaceValue analogue)
    A_face = 0.5 * (A_cell[:-1] + A_cell[1:])  # length n-1

    inv_ds2 = 1.0 / (float(ds) * float(ds))

    lower = np.zeros(n - 1, dtype=float)
    diag = np.zeros(n, dtype=float)
    upper = np.zeros(n - 1, dtype=float)

    # i=0 (left boundary, Neumann: no A_{-1/2} term)
    diag[0] = -A_face[0] * inv_ds2
    upper[0] = +A_face[0] * inv_ds2

    # interior
    for i in range(1, n - 1):
        Am = A_face[i - 1]
        Ap = A_face[i]
        lower[i - 1] = +Am * inv_ds2
        diag[i] = -(Am + Ap) * inv_ds2
        upper[i] = +Ap * inv_ds2

    # i=n-1 (right boundary)
    diag[n - 1] = -A_face[n - 2] * inv_ds2
    lower[n - 2] = +A_face[n - 2] * inv_ds2

    return lower, diag, upper


def _apply_dirichlet_tridiag(lower, diag, upper, rhs, idx, value):
    """Impose u[idx]=value in-place on a tridiagonal system."""

    n = diag.size
    idx = int(idx)
    val = float(value)

    # adjust neighbors to eliminate the fixed variable
    if idx - 1 >= 0:
        # row idx-1 has upper coefficient to u[idx]
        rhs[idx - 1] -= upper[idx - 1] * val
        upper[idx - 1] = 0.0
    if idx + 1 <= n - 1:
        # row idx+1 has lower coefficient to u[idx]
        rhs[idx + 1] -= lower[idx] * val
        lower[idx] = 0.0

    # overwrite row idx
    diag[idx] = 1.0
    rhs[idx] = val
    if idx - 1 >= 0:
        lower[idx - 1] = 0.0
    if idx <= n - 2:
        upper[idx] = 0.0


def _solve_tridiagonal_thomas(lower, diag, upper, rhs):
    """Solve a tridiagonal system using the Thomas algorithm."""

    lower = np.asarray(lower, dtype=float).copy()
    diag = np.asarray(diag, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()

    n = diag.size
    if lower.size != n - 1 or upper.size != n - 1 or rhs.size != n:
        raise ValueError("Tridiagonal shapes are inconsistent")

    # forward sweep
    for i in range(1, n):
        piv = diag[i - 1]
        if piv == 0.0 or not np.isfinite(piv):
            raise RuntimeError("Singular/ill-conditioned tridiagonal system")
        w = lower[i - 1] / piv
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]

    # back substitution
    x = np.zeros(n, dtype=float)
    piv = diag[-1]
    if piv == 0.0 or not np.isfinite(piv):
        raise RuntimeError("Singular/ill-conditioned tridiagonal system")
    x[-1] = rhs[-1] / piv
    for i in range(n - 2, -1, -1):
        piv = diag[i]
        if piv == 0.0 or not np.isfinite(piv):
            raise RuntimeError("Singular/ill-conditioned tridiagonal system")
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / piv
    return x


def solve_committor_1d_from_fes(
    s,
    F,
    D=1.0,
    beta=1.0,
    mask_q1=None,
    mask_q0=None,
):
    """Solve the 1D committor BVP: d/ds(A q') = 0 with internal Dirichlet sets.

    Parameters
    ----------
    s, F : 1D arrays
        Grid and free energy (or potential). Must be (approximately) uniform.
    D : float or 1D array
        Diffusion coefficient on the grid.
    beta : float
        Inverse thermal energy in units compatible with F.
    mask_q1, mask_q0 : boolean arrays
        Dirichlet sets for q=1 and q=0.

    Returns
    -------
    q : 1D array
        Committor values.
    """

    s = np.asarray(s, dtype=float).ravel()
    F = np.asarray(F, dtype=float).ravel()
    if s.size != F.size:
        raise ValueError("s and F must have the same length")
    ds = _require_uniform_grid_1d(s)
    n = s.size

    D = _as_1d_array(D, n, name="D")

    # conditioning shift by min(F) (does not change the solution)
    F0 = float(np.nanmin(F[np.isfinite(F)]))
    w = _safe_exp(-float(beta) * (F - F0))
    A_cell = D * w

    # Build div(A grad q) and rescale rows by 1/w to solve the better-conditioned
    # backward equation: (1/w) div(A grad q) = 0.
    lower, diag, upper = _build_tridiag_div_A_grad_1d(A_cell, ds)
    rhs = np.zeros(n, dtype=float)

    w_safe = np.maximum(w, 1e-300)
    # scale row i coefficients by 1/w_i
    diag /= w_safe
    rhs /= w_safe
    upper /= w_safe[:-1]
    lower /= w_safe[1:]

    if mask_q1 is None or mask_q0 is None:
        raise ValueError("mask_q1 and mask_q0 must be provided")

    m1 = np.asarray(mask_q1, dtype=bool).ravel()
    m0 = np.asarray(mask_q0, dtype=bool).ravel()
    if m1.size != n or m0.size != n:
        raise ValueError("mask_q1 and mask_q0 must have the same length as s")
    if np.any(m1 & m0):
        raise ValueError("mask_q1 and mask_q0 overlap")

    # Impose Dirichlet sets
    for idx in np.where(m1)[0]:
        _apply_dirichlet_tridiag(lower, diag, upper, rhs, int(idx), 1.0)
    for idx in np.where(m0)[0]:
        _apply_dirichlet_tridiag(lower, diag, upper, rhs, int(idx), 0.0)

    q = _solve_tridiagonal_thomas(lower, diag, upper, rhs)
    return np.clip(q, 0.0, 1.0)


def solve_exit_time_1d_from_fes(
    s,
    F,
    D=1.0,
    beta=1.0,
    mask_absorb=None,
):
    """Solve the 1D exit-time BVP: d/ds(A tau') = -w with tau=0 on absorbing set."""

    s = np.asarray(s, dtype=float).ravel()
    F = np.asarray(F, dtype=float).ravel()
    if s.size != F.size:
        raise ValueError("s and F must have the same length")
    ds = _require_uniform_grid_1d(s)
    n = s.size

    D = _as_1d_array(D, n, name="D")

    F0 = float(np.nanmin(F[np.isfinite(F)]))
    w = _safe_exp(-float(beta) * (F - F0))
    A_cell = D * w

    # Build div(A grad tau) and rescale rows by 1/w to solve the better-conditioned
    # backward equation: (1/w) div(A grad tau) = -1.
    lower, diag, upper = _build_tridiag_div_A_grad_1d(A_cell, ds)

    # IMPORTANT:
    # In divergence form the exit-time equation reads:
    #     d/ds(A(s) * d tau/ds) = -w(s),    with  A(s)=D(s)*w(s),  w(s)=exp(-beta(F-F0))
    # For numerical conditioning we solve the *row-scaled* system:
    #     (1/w) d/ds(A * d tau/ds) = -1
    # Therefore the RHS must be -w *before* scaling; after scaling it becomes -1.
    # A previous implementation used rhs=-1 and then divided by w, which turns the RHS
    # into -1/w and makes tau blow up in high-free-energy regions (w -> 0).
    w_safe = np.maximum(w, 1e-300)
    rhs = -w_safe.copy()

    diag /= w_safe
    rhs /= w_safe
    upper /= w_safe[:-1]
    lower /= w_safe[1:]

    if mask_absorb is None:
        raise ValueError("mask_absorb must be provided")
    ma = np.asarray(mask_absorb, dtype=bool).ravel()
    if ma.size != n:
        raise ValueError("mask_absorb must have the same length as s")

    for idx in np.where(ma)[0]:
        _apply_dirichlet_tridiag(lower, diag, upper, rhs, int(idx), 0.0)

    tau = _solve_tridiagonal_thomas(lower, diag, upper, rhs)
    tau = np.where(tau < 0.0, 0.0, tau)
    return tau


def _weighted_average_1d(var, weight, mask):
    var = np.asarray(var, dtype=float).ravel()
    weight = np.asarray(weight, dtype=float).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()
    if var.size != weight.size or var.size != mask.size:
        raise ValueError("var, weight and mask must have the same length")
    w = weight[mask]
    if w.size == 0:
        return np.nan
    den = float(np.sum(w))
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return float(np.sum(var[mask] * w) / den)


def compute_ctmc_generator_fpe_1d(
    s,
    F,
    basin_network,
    D=1.0,
    beta=1.0,
    init_weight="boltzmann",
    verbose=True,
):
    """Compute a CTMC generator between 1D basins using backward Smoluchowski BVPs.

    This is the 1D analogue of :func:`compute_ctmc_generator_fpe_fipy`, but it
    uses a NumPy tridiagonal solver (no FiPy dependency).

    Parameters
    ----------
    s, F : 1D arrays
        Uniform grid and free energy/potential.
    basin_network : BasinNetwork1D
        Must provide `labels` defined on the same grid.
    D : float or 1D array
        Diffusion coefficient.
    beta : float
        Inverse thermal energy.
    init_weight : {"boltzmann", "uniform"}
        Weight for basin-averages of exit times and committors.

    Returns
    -------
    dict with keys: K, exit_mean, k_out, p_branch, basin_ids, method.
    """

    s = np.asarray(s, dtype=float).ravel()
    F = np.asarray(F, dtype=float).ravel()
    if s.size != F.size:
        raise ValueError("s and F must have the same length")
    _ = _require_uniform_grid_1d(s)

    labels_src = getattr(basin_network, "core_labels", None)
    if labels_src is None:
        labels_src = basin_network.labels
    labels = np.asarray(labels_src, dtype=int).ravel()
    if labels.size != s.size:
        raise ValueError("basin_network.labels must have the same length as s")

    basin_ids = np.unique(labels[labels >= 0])
    basin_ids = np.sort(basin_ids)
    n_basins = int(basin_ids.size)
    if n_basins < 1:
        raise ValueError("No basins found (labels>=0) — cannot build CTMC")

    in_any_basin = labels >= 0
    basin_masks = [(labels == bid) for bid in basin_ids]

    # weights for basin averages
    if str(init_weight).lower() == "uniform":
        weight = np.ones_like(F, dtype=float)
    elif str(init_weight).lower() == "boltzmann":
        F0 = float(np.nanmin(F[np.isfinite(F)]))
        weight = _safe_exp(-float(beta) * (F - F0))
    else:
        raise ValueError("init_weight must be 'boltzmann' or 'uniform'")

    exit_mean = np.full(n_basins, np.nan, dtype=float)
    k_out = np.full(n_basins, np.nan, dtype=float)
    p_branch = np.full((n_basins, n_basins), np.nan, dtype=float)
    K = np.zeros((n_basins, n_basins), dtype=float)

    if verbose:
        print(f"[1D CTMC] n_basins={n_basins}, init_weight={init_weight}")

    # --- exit times ---
    for i in range(n_basins):
        mask_i = basin_masks[i]
        if not np.any(mask_i):
            continue
        mask_abs = in_any_basin & (~mask_i)

        tau = solve_exit_time_1d_from_fes(s=s, F=F, D=D, beta=beta, mask_absorb=mask_abs)
        tau_mean = _weighted_average_1d(tau, weight, mask_i)
        exit_mean[i] = float(tau_mean) if np.isfinite(tau_mean) else np.nan
        if np.isfinite(exit_mean[i]) and exit_mean[i] > 0:
            k_out[i] = 1.0 / exit_mean[i]
        if verbose:
            print(f"[1D CTMC] basin {i}: <tau_exit>={exit_mean[i]:.6g}, k_out={k_out[i]:.6g}")

    # --- committors and branching ---
    if n_basins == 2:
        for i in range(n_basins):
            if not np.isfinite(k_out[i]) or k_out[i] <= 0:
                continue
            j = 1 - i
            p_branch[i, j] = 1.0
            K[i, j] = float(k_out[i])
            K[i, i] = -float(k_out[i])
        if verbose:
            print("[1D CTMC] n_basins=2 -> setting p_branch=1 and K_ij=k_out(i) by construction")
    else:
        for i in range(n_basins):
            if not np.isfinite(k_out[i]) or k_out[i] <= 0:
                continue
            mask_i = basin_masks[i]

            for j in range(n_basins):
                if i == j:
                    continue
                mask_j = basin_masks[j]
                if not np.any(mask_j):
                    continue

                mask_q1 = mask_j
                mask_q0 = in_any_basin & (~mask_i) & (~mask_j)
                if not np.any(mask_q0):
                    p_branch[i, j] = 1.0
                    continue

                q = solve_committor_1d_from_fes(s=s, F=F, D=D, beta=beta, mask_q1=mask_q1, mask_q0=mask_q0)
                pij = _weighted_average_1d(q, weight, mask_i)
                p_branch[i, j] = float(np.clip(pij, 0.0, 1.0)) if np.isfinite(pij) else np.nan

            row = p_branch[i, :].copy()
            row[i] = np.nan
            srow = np.nansum(row)
            if not (np.isfinite(srow) and srow > 0):
                raise RuntimeError(
                    f"1D CTMC: branching probabilities are all zero for basin {i}. "
                    "This usually indicates a labeling or mask construction issue."
                )
            for j in range(n_basins):
                if i == j:
                    continue
                if np.isfinite(p_branch[i, j]):
                    p_branch[i, j] /= float(srow)

            for j in range(n_basins):
                if i == j:
                    continue
                if np.isfinite(p_branch[i, j]) and p_branch[i, j] > 0:
                    K[i, j] = float(k_out[i] * p_branch[i, j])
            K[i, i] = -float(np.sum(K[i, :]))

            if verbose:
                pr = p_branch[i, :].copy(); pr[i] = 0.0
                print(f"[1D CTMC] basin {i}: sum p_branch={np.nansum(pr):.6g}, row={pr}")

    if verbose:
        rs = K.sum(axis=1)
        print(f"[1D CTMC] row-sum check (should be ~0): min={rs.min():.3e}, max={rs.max():.3e}")

    return {
        "K": K,
        "exit_mean": exit_mean,
        "k_out": k_out,
        "p_branch": p_branch,
        "basin_ids": np.asarray(basin_ids, dtype=int),
        "method": "numpy_backward_ctmc_1d",
    }


# =============================================================================
# Backward FPE kinetics with FiPy (CTMC generator from basins)
# =============================================================================

def _infer_edges_from_centers_1d(centers):
    """Infer cell-edge coordinates from a 1D array of cell centers."""

    c = np.asarray(centers, dtype=float).ravel()
    if c.size < 2:
        raise ValueError("Need at least two centers to infer edges")
    if not np.all(np.diff(c) > 0):
        raise ValueError("Centers must be strictly increasing")

    edges = np.empty(c.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - (edges[1] - c[0])
    edges[-1] = c[-1] + (c[-1] - edges[-2])
    return edges


def _is_uniform_spacing(edges, rtol=1e-10, atol=1e-12):
    e = np.asarray(edges, dtype=float).ravel()
    if e.size < 3:
        return True
    d = np.diff(e)
    return bool(np.allclose(d, d.mean(), rtol=rtol, atol=atol))


def build_mesh_2d_from_edges(x_edges, y_edges, overlap=2):
    """Build a FiPy 2D rectangular mesh from 1D edge arrays."""

    if not _HAVE_FIPY:
        raise ImportError("FiPy is required for FiPy-based kinetics solvers")

    xe = np.asarray(x_edges, dtype=float).ravel()
    ye = np.asarray(y_edges, dtype=float).ravel()
    if xe.size < 2 or ye.size < 2:
        raise ValueError("Edge arrays must have length >= 2")
    if not np.all(np.diff(xe) > 0) or not np.all(np.diff(ye) > 0):
        raise ValueError("Edge arrays must be strictly increasing")

    dx = np.diff(xe)
    dy = np.diff(ye)
    nx = int(dx.size)
    ny = int(dy.size)

    if _is_uniform_spacing(xe) and _is_uniform_spacing(ye):
        mesh = Grid2D(nx=nx, ny=ny, dx=float(dx.mean()), dy=float(dy.mean()), overlap=int(overlap))
    else:
        if NonUniformGrid2D is None:
            raise ImportError(
                "NonUniformGrid2D is not available in this FiPy installation; "
                "please provide uniform edges or upgrade FiPy"
            )
        mesh = NonUniformGrid2D(nx=nx, ny=ny, dx=dx, dy=dy, overlap=int(overlap))

    mesh = mesh + ((float(xe[0]),), (float(ye[0]),))

    return mesh, nx, ny, dx, dy


def _fipy_cells_to_ij_from_edges(mesh, x_edges, y_edges):
    """Map FiPy cells to (i,j) indices on the (nx,ny) grid defined by edges."""

    x_edges = np.asarray(x_edges, dtype=float).ravel()
    y_edges = np.asarray(y_edges, dtype=float).ravel()
    nx = x_edges.size - 1
    ny = y_edges.size - 1

    cx = np.asarray(numerix.array(mesh.cellCenters[0]), dtype=float)
    cy = np.asarray(numerix.array(mesh.cellCenters[1]), dtype=float)

    i = np.searchsorted(x_edges, cx, side="right") - 1
    j = np.searchsorted(y_edges, cy, side="right") - 1
    i = np.clip(i, 0, nx - 1)
    j = np.clip(j, 0, ny - 1)

    xc = 0.5 * (x_edges[i] + x_edges[i + 1])
    yc = 0.5 * (y_edges[j] + y_edges[j + 1])
    dx = np.maximum(np.abs(x_edges[i + 1] - x_edges[i]), 1e-12)
    dy = np.maximum(np.abs(y_edges[j + 1] - y_edges[j]), 1e-12)

    err = float(np.nanmax(np.maximum(np.abs(cx - xc) / dx, np.abs(cy - yc) / dy)))
    if (not np.isfinite(err)) or err > 0.25:
        raise RuntimeError(
            "FiPy CTMC: failed to map FiPy cells to (i,j) indices from edges. "
            f"max normalized center error={err}."
        )

    return i.astype(int), j.astype(int)


def _ij_field_to_fipy_cells(field_ij, i_of_cell, j_of_cell, dtype=float):
    """Map a (nx,ny) array to a FiPy cell-ordered 1D array using i/j index arrays."""

    a = np.asarray(field_ij)
    if a.ndim != 2:
        raise ValueError("Expected a 2D array (nx, ny)")
    return np.asarray(a[i_of_cell, j_of_cell], dtype=dtype)


def make_backward_coefficient_A_face_from_cell_values(mesh, U_cell, D_cell, beta, name_U="U", name_D="D"):
    """Construct w=exp(-beta U) and A_face = (D*w).arithmeticFaceValue."""

    if not _HAVE_FIPY:
        raise ImportError("FiPy is required for FiPy-based kinetics solvers")

    U_cell = np.asarray(U_cell, dtype=float).ravel()
    if U_cell.size != mesh.numberOfCells:
        raise ValueError("U_cell must be one value per FiPy cell")

    # ---- sanitize U_cell to avoid NaN/inf in FiPy coefficients ----
    finiteU = np.isfinite(U_cell)
    if not np.any(finiteU):
        raise ValueError("U_cell has no finite values")
    U_min_finite = float(np.min(U_cell[finiteU]))
    if float(beta) <= 0:
        raise ValueError("beta must be > 0")
    # Fill non-finite energies with a high barrier (in U units)
    # Barrier height chosen in kT units for numerical stability.
    barrier_kT = 50.0
    U_fill = U_min_finite + float(barrier_kT) / float(beta)
    if not np.all(finiteU):
        U_cell = U_cell.copy()
        U_cell[~finiteU] = U_fill

    U_var = CellVariable(mesh=mesh, name=name_U, value=U_cell)

    # Shift by min(U) for conditioning: exp(-beta(U-Umin)) has max 1.
    U_min = float(np.min(U_cell))
    w_var = numerix.exp(-float(beta) * (U_var - U_min))

    if np.isscalar(D_cell):
        D_scalar = float(D_cell)
        if (not np.isfinite(D_scalar)) or (D_scalar <= 0.0):
            D_scalar = 1e-12
        D_var = D_scalar
        A_cell = D_scalar * w_var
    else:
        D_cell = np.asarray(D_cell, dtype=float).ravel()
        if D_cell.size != mesh.numberOfCells:
            raise ValueError("D_cell must be one value per FiPy cell")

        # Sanitize D: FiPy does not like NaN/inf and negative diffusion
        badD = (~np.isfinite(D_cell)) | (D_cell <= 0)
        if np.any(badD):
            D_cell = D_cell.copy()
            D_cell[badD] = 1e-12

        D_var = CellVariable(mesh=mesh, name=name_D, value=D_cell)
        A_cell = D_var * w_var

    A_face = A_cell.arithmeticFaceValue
    return U_var, w_var, A_face, D_var


def _internal_dirichlet_penalty(var, mask_cells, value, large_value):
    """Return (implicit_term, rhs_add) to enforce an internal Dirichlet constraint."""

    m = numerix.array(mask_cells, dtype=float).ravel()
    n = int(var.mesh.numberOfCells)
    if m.size != n:
        raise ValueError(f"mask_cells length {m.size} != mesh.numberOfCells {n}")

    lv = float(large_value)
    mask_var = CellVariable(mesh=var.mesh, name="_dirichlet_mask", value=m)
    pen = lv * mask_var

    return ImplicitSourceTerm(coeff=pen, var=var), (pen * float(value))


def _safe_scalar_max(coeff) -> float:
    """Robust scalar max() for FiPy coefficients."""

    if isinstance(coeff, (tuple, list)):
        return float(max(_safe_scalar_max(c) for c in coeff))

    if hasattr(coeff, "value"):
        coeff = coeff.value

    arr = np.asarray(coeff, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(arr))


def solve_committor_fipy(
    mesh,
    A_face,
    mask_q1,
    mask_q0,
    large_value=None,
    enforce_reflecting=True,
    solver=None,
    sweep_tol=1e-10,
    max_sweeps=2000,
    verbose=False,
):
    """Solve the 2D committor BVP on a FiPy mesh.

    Solves :math:`\\nabla\\cdot(A\\,\\nabla q) = 0` with internal
    Dirichlet constraints :math:`q=1` (*mask_q1*) and :math:`q=0`
    (*mask_q0*), enforced via large penalty terms.

    Parameters
    ----------
    mesh : FiPy mesh
        2D mesh.
    A_face : FiPy FaceVariable
        Backward coefficient :math:`A = D\\,e^{-\\beta U}` on faces.
    mask_q1, mask_q0 : array_like of bool
        Cell masks for the two Dirichlet sets.
    large_value : float or None
        Penalty magnitude (auto-scaled if ``None``).
    enforce_reflecting : bool
        Impose no-flux on exterior faces.
    solver, sweep_tol, max_sweeps : optional
        FiPy solver parameters.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    q : CellVariable
        Committor field.
    info : dict
        ``{'residual', 'n_sweeps'}``.
    """

    if not _HAVE_FIPY:
        raise ImportError("FiPy is required for FiPy-based kinetics solvers")

    m1 = numerix.array(mask_q1, dtype=bool)
    m0 = numerix.array(mask_q0, dtype=bool)

    q = CellVariable(mesh=mesh, name="committor", value=0.0)

    if large_value is None:
        large_value = max(1e6, 1e4 * float(_safe_scalar_max(A_face)))

    if solver is None:
        solver = _default_fipy_solver(tolerance=float(sweep_tol), iterations=int(max_sweeps))

    if enforce_reflecting:
        q.faceGrad.constrain(((0.0,), (0.0,)), where=mesh.exteriorFaces)

    lhs = DiffusionTerm(coeff=A_face, var=q)
    imp1, rhs1 = _internal_dirichlet_penalty(q, m1, 1.0, large_value)
    imp0, rhs0 = _internal_dirichlet_penalty(q, m0, 0.0, large_value)

    eq = (lhs + imp1 + imp0 == rhs1 + rhs0)

    eq.solve(var=q, solver=solver)

    residual = float("nan")
    n_sweeps = 1

    if verbose:
        print(f"[FiPy committor] sweeps={n_sweeps}, residual={residual}")

    return q, {"residual": residual, "n_sweeps": n_sweeps}


def solve_exit_time_fipy(
    mesh,
    A_face,
    w_var,
    mask_absorb,
    large_value=None,
    enforce_reflecting=True,
    solver=None,
    sweep_tol=1e-10,
    max_sweeps=4000,
    verbose=False,
):
    """Solve the 2D mean exit-time BVP on a FiPy mesh.

    Solves :math:`\\nabla\\cdot(A\\,\\nabla\\tau) = -w` with
    :math:`\\tau=0` on absorbing cells, enforced via penalty.

    Parameters
    ----------
    mesh : FiPy mesh
        2D mesh.
    A_face : FiPy FaceVariable
        Backward coefficient on faces.
    w_var : CellVariable
        Boltzmann weight :math:`w = e^{-\\beta(U-U_{\\min})}`.
    mask_absorb : array_like of bool
        Absorbing cell mask.
    large_value : float or None
        Penalty magnitude.
    enforce_reflecting : bool
        Impose no-flux on exterior faces.
    solver, sweep_tol, max_sweeps : optional
        FiPy solver parameters.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    tau : CellVariable
        Mean exit-time field.
    info : dict
        ``{'residual', 'n_sweeps'}``.
    """

    if not _HAVE_FIPY:
        raise ImportError("FiPy is required for FiPy-based kinetics solvers")

    ma = numerix.array(mask_absorb, dtype=bool)
    tau = CellVariable(mesh=mesh, name="exit_time", value=0.0)

    if large_value is None:
        large_value = max(1e6, 1e4 * float(_safe_scalar_max(A_face)))

    if solver is None:
        solver = _default_fipy_solver(tolerance=float(sweep_tol), iterations=int(max_sweeps))

    if enforce_reflecting:
        tau.faceGrad.constrain(((0.0,), (0.0,)), where=mesh.exteriorFaces)

    lhs = DiffusionTerm(coeff=A_face, var=tau)
    imp, rhs_add = _internal_dirichlet_penalty(tau, ma, 0.0, large_value)

    eq = (lhs + imp == -w_var + rhs_add)

    residual = None
    n_sweeps = 1

    try:
        eq.solve(var=tau, solver=solver)
    except Exception:
        residual = np.inf
        n_sweeps = 0
        for n_sweeps in range(1, int(max_sweeps) + 1):
            residual = float(eq.sweep(var=tau, solver=solver))
            if residual < float(sweep_tol):
                break

    if verbose:
        print(f"[FiPy exit-time] sweeps={n_sweeps}, residual={residual}")

    return tau, {"residual": residual, "n_sweeps": n_sweeps}


def _weighted_basin_average(mesh, var_cell, weight_cell, basin_mask):
    """Volume-weighted average of var_cell over basin_mask with weights weight_cell."""

    vol = numerix.array(mesh.cellVolumes, dtype=float)
    v = numerix.array(var_cell, dtype=float)
    w = numerix.array(weight_cell, dtype=float)
    m = numerix.array(basin_mask, dtype=float)

    num = float((v * w * vol * m).sum())
    den = float((w * vol * m).sum())
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return num / den


def compute_ctmc_generator_fpe_fipy(
    x_centers,
    y_centers,
    U_ij,
    basin_network,
    D=1.0,
    beta=1.0,
    x_edges=None,
    y_edges=None,
    init_weight="boltzmann",
    large_value=None,
    enforce_reflecting=True,
    solver=None,
    sweep_tol=1e-10,
    verbose=True,
):
    r"""Compute a CTMC generator between basins using backward FPE FiPy solves.

    For each basin *i*:

    1. Solve exit time :math:`\tau_i` to the union of other basins (absorbing).
       :math:`k_{\text{out}}(i) = 1 / \langle\tau_i\rangle` (average over basin *i*).
    2. For each :math:`j \neq i`, solve committor :math:`q_{ij}` with
       :math:`q=1` on basin *j* and :math:`q=0` on all other basins.
       :math:`p_{ij} = \langle q_{ij}\rangle` averaged over basin *i*.
    3. Set :math:`K_{ij} = k_{\text{out}}(i)\, p_{ij}` and
       :math:`K_{ii} = -\sum_{j \neq i} K_{ij}`.

    Returns
    -------
    dict with keys: K, exit_mean, k_out, p_branch, basin_ids, mesh, info, method.
    """

    if not _HAVE_FIPY:
        raise ImportError("FiPy is required for compute_ctmc_generator_fpe_fipy()")

    x = np.asarray(x_centers, dtype=float).ravel()
    y = np.asarray(y_centers, dtype=float).ravel()
    U = np.asarray(U_ij, dtype=float)

    if U.shape != (x.size, y.size):
        raise ValueError(f"U_ij shape {U.shape} must match (nx,ny)=({x.size},{y.size})")

    labels_src = getattr(basin_network, 'core_labels', None)
    if labels_src is None:
        labels_src = basin_network.labels
    labels = np.asarray(labels_src, dtype=int)
    if labels.shape != U.shape:
        raise ValueError("basin_network.labels must have the same shape as U_ij")

    # Basin IDs from labels (do not assume contiguous 0..n_basins-1)
    basin_ids = np.unique(labels[labels >= 0])
    basin_ids = np.sort(basin_ids)
    n_basins = int(basin_ids.size)

    if n_basins < 1:
        raise ValueError("No basins found (labels>=0) — cannot build CTMC")

    if x_edges is None:
        x_edges = _infer_edges_from_centers_1d(x)
    if y_edges is None:
        y_edges = _infer_edges_from_centers_1d(y)

    mesh, nx, ny, dx, dy = build_mesh_2d_from_edges(x_edges, y_edges)

    # Robust mapping FiPy cells -> (i,j) on the provided (nx,ny) grid
    i_of_cell, j_of_cell = _fipy_cells_to_ij_from_edges(mesh, x_edges, y_edges)

    # Map fields to FiPy cell ordering
    U_cell = _ij_field_to_fipy_cells(U, i_of_cell, j_of_cell, dtype=float)

    if callable(D):
        cx = np.asarray(mesh.cellCenters[0], dtype=float)
        cy = np.asarray(mesh.cellCenters[1], dtype=float)
        D_cell = np.asarray(D(cx, cy), dtype=float).ravel()
        if D_cell.size != mesh.numberOfCells:
            raise ValueError("Callable D(x,y) must return one value per cell")
    elif np.isscalar(D):
        D_cell = float(D)
    else:
        D_cell = _ij_field_to_fipy_cells(np.asarray(D, dtype=float), i_of_cell, j_of_cell, dtype=float)

    # Backward coefficient A = D * exp(-beta U)
    U_var, w_var, A_face, D_var = make_backward_coefficient_A_face_from_cell_values(
        mesh, U_cell, D_cell, beta
    )

    # weights for basin averages
    if init_weight.lower() == "uniform":
        weight_cell = numerix.ones(mesh.numberOfCells, dtype=float)
    elif init_weight.lower() == "boltzmann":
        weight_cell = w_var
    else:
        raise ValueError("init_weight must be 'boltzmann' or 'uniform'")

    labels_cell = _ij_field_to_fipy_cells(labels, i_of_cell, j_of_cell, dtype=int)
    basin_masks = [(labels_cell == bid) for bid in basin_ids]
    in_any_basin = (labels_cell >= 0)

    exit_mean = np.full(n_basins, np.nan, dtype=float)
    k_out = np.full(n_basins, np.nan, dtype=float)
    p_branch = np.full((n_basins, n_basins), np.nan, dtype=float)
    K = np.zeros((n_basins, n_basins), dtype=float)

    info = {"exit": {}, "committor": {}}

    if verbose:
        lv_str = "None" if large_value is None else f"{float(large_value):g}"
        print(f"[FiPy CTMC] n_basins={n_basins}, init_weight={init_weight}, large_value={lv_str}")

    # --- exit time per basin ---
    for i in range(n_basins):
        mask_i = basin_masks[i]
        if not np.any(mask_i):
            if verbose:
                print(f"[FiPy CTMC] basin {i}: empty mask -> skipping")
            continue

        # absorbing = all other basins
        mask_abs = in_any_basin & (~mask_i)

        tau, inf_tau = solve_exit_time_fipy(
            mesh=mesh,
            A_face=A_face,
            w_var=w_var,
            mask_absorb=mask_abs,
            large_value=large_value,
            enforce_reflecting=enforce_reflecting,
            solver=solver,
            sweep_tol=sweep_tol,
            verbose=verbose,
        )
        info["exit"][int(i)] = inf_tau

        tau_mean = _weighted_basin_average(mesh, tau, weight_cell, mask_i)
        exit_mean[i] = float(tau_mean) if np.isfinite(tau_mean) else np.nan
        if np.isfinite(exit_mean[i]) and exit_mean[i] > 0:
            k_out[i] = 1.0 / exit_mean[i]

        if verbose:
            print(f"[FiPy CTMC] basin {i}: <tau_exit>={exit_mean[i]:.6g}, k_out={k_out[i]:.6g}")

    # --- committors and branching ---
    if n_basins == 2:
        for i in range(n_basins):
            if not np.isfinite(k_out[i]) or k_out[i] <= 0:
                continue
            j = 1 - i
            p_branch[i, j] = 1.0
            K[i, j] = float(k_out[i])
            K[i, i] = -float(k_out[i])
        if verbose:
            print("[FiPy CTMC] n_basins=2 -> setting p_branch=1 and K_ij=k_out(i) by construction")
    else:
        for i in range(n_basins):
            if not np.isfinite(k_out[i]) or k_out[i] <= 0:
                continue

            mask_i = basin_masks[i]

            for j in range(n_basins):
                if i == j:
                    continue

                mask_j = basin_masks[j]
                if not np.any(mask_j):
                    continue

                mask_q1 = mask_j
                mask_q0 = in_any_basin & (~mask_i) & (~mask_j)

                if np.any(mask_i & mask_q0) or np.any(mask_i & mask_q1) or np.any(mask_q0 & mask_q1):
                    raise RuntimeError(
                        f"FiPy CTMC: committor masks overlap for (i={i}, j={j}). "
                        "This indicates a label→cell mapping or labeling bug."
                    )

                if not np.any(mask_q0):
                    p_branch[i, j] = 1.0
                    continue

                q, inf_q = solve_committor_fipy(
                    mesh=mesh,
                    A_face=A_face,
                    mask_q1=mask_q1,
                    mask_q0=mask_q0,
                    large_value=large_value,
                    enforce_reflecting=enforce_reflecting,
                    solver=solver,
                    sweep_tol=sweep_tol,
                    verbose=False,
                )
                info["committor"][(int(i), int(j))] = inf_q

                pij = _weighted_basin_average(mesh, q, weight_cell, mask_i)
                if np.isfinite(pij):
                    p_branch[i, j] = float(np.clip(float(pij), 0.0, 1.0))
                else:
                    p_branch[i, j] = np.nan

            # normalize branching row defensively
            row = p_branch[i, :].copy()
            row[i] = np.nan
            srow = np.nansum(row)
            if not (np.isfinite(srow) and srow > 0):
                if verbose:
                    print(
                        f"[FiPy CTMC][ERROR] basin {i}: sum p_branch=0. "
                        "Cannot build a meaningful CTMC row. "
                        "Check committor masks and label→FiPy mapping."
                    )
                raise RuntimeError(
                    f"FiPy CTMC: branching probabilities are all zero for basin {i}. "
                    "This usually means a committor mask bug (start basin clamped to 0) "
                    "or a mismatch between BasinNetwork labels and FiPy cell ordering."
                )

            for j in range(n_basins):
                if i == j:
                    continue
                if np.isfinite(p_branch[i, j]):
                    p_branch[i, j] /= float(srow)

            if verbose:
                pr = p_branch[i, :].copy()
                pr[i] = 0.0
                print(f"[FiPy CTMC] basin {i}: sum p_branch={np.nansum(pr):.6g}, row={pr}")

            for j in range(n_basins):
                if i == j:
                    continue
                if np.isfinite(p_branch[i, j]) and p_branch[i, j] > 0:
                    K[i, j] = float(k_out[i] * p_branch[i, j])

            K[i, i] = -float(np.sum(K[i, :]))

    if verbose:
        rs = K.sum(axis=1)
        print(f"[FiPy CTMC] row-sum check (should be ~0): min={rs.min():.3e}, max={rs.max():.3e}")

    return {
        "K": K,
        "exit_mean": exit_mean,
        "k_out": k_out,
        "p_branch": p_branch,
        "mesh": mesh,
        "info": info,
        "basin_ids": np.asarray(basin_ids, dtype=int),
        "method": "fipy_backward_ctmc",
        "nx": nx,
        "ny": ny,
        "x_edges": np.asarray(x_edges, dtype=float),
        "y_edges": np.asarray(y_edges, dtype=float),
    }
