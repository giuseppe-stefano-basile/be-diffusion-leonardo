"""Smooth positive one-dimensional diffusion models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class SplineLogDiffusion1D:
    """Cubic B-spline model for ``log D(s)``.

    ``D(s)`` is positive by construction through exponentiation.  Optional
    clipping is applied after exponentiation; derivatives are set to zero in
    clipped regions to avoid artificial drift spikes at the imposed bounds.
    """

    s_min: float
    s_max: float
    n_basis: int = 20
    degree: int = 3
    D_min: float = 1e-10
    D_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.s_max <= self.s_min:
            raise ValueError("s_max must be larger than s_min")
        if self.n_basis < self.degree + 1:
            raise ValueError("n_basis must be at least degree + 1")
        if self.D_min <= 0:
            raise ValueError("D_min must be positive")
        if self.D_max is not None and self.D_max <= self.D_min:
            raise ValueError("D_max must be larger than D_min")
        self.knots = self._make_knots()

    def _make_knots(self) -> np.ndarray:
        internal_count = self.n_basis - self.degree - 1
        if internal_count > 0:
            internal = np.linspace(self.s_min, self.s_max, internal_count + 2)[1:-1]
        else:
            internal = np.array([], dtype=float)
        return np.r_[
            np.full(self.degree + 1, self.s_min, dtype=float),
            internal,
            np.full(self.degree + 1, self.s_max, dtype=float),
        ]

    def basis_matrix(self, s, derivative: int = 0) -> np.ndarray:
        """Evaluate the spline basis or its derivative at positions ``s``."""

        x = np.asarray(s, dtype=float).ravel()
        x_eval = np.clip(x, self.s_min, self.s_max)
        out = np.empty((x_eval.size, self.n_basis), dtype=float)
        eye = np.eye(self.n_basis)
        for i in range(self.n_basis):
            spline = BSpline(self.knots, eye[i], self.degree, extrapolate=False)
            if derivative:
                spline = spline.derivative(derivative)
            vals = spline(x_eval)
            out[:, i] = np.where(np.isfinite(vals), vals, 0.0)
        return out

    def logD(self, s, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        if coeffs.size != self.n_basis:
            raise ValueError("coeffs length does not match n_basis")
        return self.basis_matrix(s) @ coeffs

    def dlogD(self, s, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        return self.basis_matrix(s, derivative=1) @ coeffs

    def d2logD(self, s, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        return self.basis_matrix(s, derivative=2) @ coeffs

    def evaluate(self, s, coeffs: np.ndarray) -> np.ndarray:
        raw = np.exp(np.clip(self.logD(s, coeffs), -700.0, 700.0))
        upper = np.inf if self.D_max is None else float(self.D_max)
        return np.clip(raw, float(self.D_min), upper)

    def derivative(self, s, coeffs: np.ndarray) -> np.ndarray:
        logd = self.logD(s, coeffs)
        raw = np.exp(np.clip(logd, -700.0, 700.0))
        d_raw = raw * self.dlogD(s, coeffs)
        clipped_low = raw <= self.D_min
        clipped_high = np.zeros_like(clipped_low, dtype=bool)
        if self.D_max is not None:
            clipped_high = raw >= self.D_max
        return np.where(clipped_low | clipped_high, 0.0, d_raw)

    def roughness_penalty(
        self,
        coeffs: np.ndarray,
        grid: np.ndarray,
        *,
        smooth_lambda: float = 0.0,
        curvature_lambda: float = 0.0,
    ) -> float:
        grid = np.asarray(grid, dtype=float).ravel()
        penalty = 0.0
        integrate = getattr(np, "trapezoid", np.trapz)
        if smooth_lambda:
            d1 = self.dlogD(grid, coeffs)
            penalty += float(smooth_lambda) * float(integrate(d1 * d1, grid))
        if curvature_lambda:
            d2 = self.d2logD(grid, coeffs)
            penalty += float(curvature_lambda) * float(integrate(d2 * d2, grid))
        return float(penalty)


@dataclass
class GridLogDiffusion1D:
    """Piecewise-linear control-grid model for ``log D(s)``.

    This is intentionally simpler and more local than a cubic spline basis:
    the parameters are log-diffusion values on a uniform control grid and
    values between controls are linearly interpolated.  Smoothness is imposed
    through finite-difference penalties, which makes the model useful as a
    controlled P-spline/grid-logD alternative for finite-lag transition fits.
    """

    s_min: float
    s_max: float
    n_nodes: int = 25
    D_min: float = 1e-10
    D_max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.s_max <= self.s_min:
            raise ValueError("s_max must be larger than s_min")
        if int(self.n_nodes) < 2:
            raise ValueError("n_nodes must be at least 2")
        if self.D_min <= 0:
            raise ValueError("D_min must be positive")
        if self.D_max is not None and self.D_max <= self.D_min:
            raise ValueError("D_max must be larger than D_min")
        self.n_nodes = int(self.n_nodes)
        self.nodes = np.linspace(float(self.s_min), float(self.s_max), self.n_nodes)

    @property
    def n_basis(self) -> int:
        return int(self.n_nodes)

    def logD(self, s, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        if coeffs.size != self.n_nodes:
            raise ValueError("coeffs length does not match n_nodes")
        x = np.asarray(s, dtype=float).ravel()
        x_eval = np.clip(x, self.s_min, self.s_max)
        return np.interp(x_eval, self.nodes, coeffs)

    def dlogD(self, s, coeffs: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        if coeffs.size != self.n_nodes:
            raise ValueError("coeffs length does not match n_nodes")
        slopes = np.diff(coeffs) / np.diff(self.nodes)
        x = np.asarray(s, dtype=float).ravel()
        idx = np.searchsorted(self.nodes, np.clip(x, self.s_min, self.s_max), side="right") - 1
        idx = np.clip(idx, 0, slopes.size - 1)
        return slopes[idx]

    def evaluate(self, s, coeffs: np.ndarray) -> np.ndarray:
        raw = np.exp(np.clip(self.logD(s, coeffs), -700.0, 700.0))
        upper = np.inf if self.D_max is None else float(self.D_max)
        return np.clip(raw, float(self.D_min), upper)

    def derivative(self, s, coeffs: np.ndarray) -> np.ndarray:
        logd = self.logD(s, coeffs)
        raw = np.exp(np.clip(logd, -700.0, 700.0))
        d_raw = raw * self.dlogD(s, coeffs)
        clipped_low = raw <= self.D_min
        clipped_high = np.zeros_like(clipped_low, dtype=bool)
        if self.D_max is not None:
            clipped_high = raw >= self.D_max
        return np.where(clipped_low | clipped_high, 0.0, d_raw)

    def roughness_penalty(
        self,
        coeffs: np.ndarray,
        grid: np.ndarray = None,
        *,
        smooth_lambda: float = 0.0,
        curvature_lambda: float = 0.0,
    ) -> float:
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        if coeffs.size != self.n_nodes:
            raise ValueError("coeffs length does not match n_nodes")
        penalty = 0.0
        if smooth_lambda and coeffs.size >= 2:
            d1 = np.diff(coeffs)
            penalty += float(smooth_lambda) * float(np.mean(d1 * d1))
        if curvature_lambda and coeffs.size >= 3:
            d2 = np.diff(coeffs, n=2)
            penalty += float(curvature_lambda) * float(np.mean(d2 * d2))
        return float(penalty)
