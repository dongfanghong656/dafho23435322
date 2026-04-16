import argparse
import ctypes
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks


def mie_ab(m, x, nmax=None):
    if x < 1e-12:
        nmax = nmax or 5
        return np.zeros(nmax, dtype=complex), np.zeros(nmax, dtype=complex)
    if nmax is None:
        nmax = max(int(x + 4.05 * x ** (1.0 / 3.0) + 2) + 2, 5)
    mx = m * x
    nmx = max(nmax + 1, int(abs(mx)) + 1) + 20
    d = np.zeros(nmx + 2, dtype=complex)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)
    psi = np.zeros(nmax + 2)
    chi = np.zeros(nmax + 2)
    psi[0], psi[1] = np.sin(x), np.sin(x) / x - np.cos(x)
    chi[0], chi[1] = np.cos(x), np.cos(x) / x + np.sin(x)
    for n in range(1, nmax + 1):
        psi[n + 1] = (2 * n + 1) / x * psi[n] - psi[n - 1]
        chi[n + 1] = (2 * n + 1) / x * chi[n] - chi[n - 1]
    xi = psi - 1j * chi
    a = np.zeros(nmax, dtype=complex)
    b = np.zeros(nmax, dtype=complex)
    for n in range(1, nmax + 1):
        a[n - 1] = ((d[n] / m + n / x) * psi[n] - psi[n - 1]) / ((d[n] / m + n / x) * xi[n] - xi[n - 1])
        b[n - 1] = ((m * d[n] + n / x) * psi[n] - psi[n - 1]) / ((m * d[n] + n / x) * xi[n] - xi[n - 1])
    return a, b


def s_back_full(a, b):
    return 0.5 * sum((2 * n + 1) * ((-1) ** n) * (a[n - 1] - b[n - 1]) for n in range(1, len(a) + 1))


def n_tio2_anatase(l_um):
    return np.sqrt(5.825 + 0.2441 / (l_um**2 - 0.0803))


_FE2O3_O_DATA = np.array(
    [
        [0.70, 2.972, 0.031], [0.71, 2.956, 0.028], [0.72, 2.942, 0.026], [0.73, 2.929, 0.024],
        [0.74, 2.916, 0.022], [0.75, 2.903, 0.021], [0.76, 2.892, 0.020], [0.77, 2.882, 0.020],
        [0.78, 2.872, 0.019], [0.79, 2.862, 0.019], [0.80, 2.853, 0.020], [0.81, 2.845, 0.022],
        [0.82, 2.838, 0.024], [0.83, 2.833, 0.025], [0.84, 2.828, 0.026], [0.85, 2.824, 0.027],
        [0.86, 2.820, 0.027], [0.87, 2.816, 0.026], [0.88, 2.813, 0.026], [0.89, 2.809, 0.026],
        [0.90, 2.805, 0.024], [0.91, 2.801, 0.024], [0.92, 2.798, 0.023], [0.93, 2.794, 0.023],
        [0.94, 2.791, 0.023], [0.95, 2.789, 0.022], [0.96, 2.787, 0.021], [0.97, 2.784, 0.020],
        [0.98, 2.781, 0.018], [0.99, 2.778, 0.017], [1.00, 2.775, 0.015], [1.01, 2.771, 0.015],
        [1.02, 2.768, 0.014], [1.03, 2.765, 0.013], [1.04, 2.762, 0.012], [1.05, 2.759, 0.011],
        [1.06, 2.755, 0.011], [1.07, 2.753, 0.011], [1.08, 2.750, 0.011], [1.09, 2.747, 0.011],
        [1.10, 2.745, 0.011],
    ]
)
_FE2O3_E_DATA = np.array(
    [
        [0.70, 2.675, 0.075], [0.71, 2.662, 0.072], [0.72, 2.652, 0.068], [0.73, 2.641, 0.066],
        [0.74, 2.631, 0.063], [0.75, 2.621, 0.061], [0.76, 2.612, 0.060], [0.77, 2.604, 0.059],
        [0.78, 2.596, 0.058], [0.79, 2.589, 0.057], [0.80, 2.582, 0.057], [0.81, 2.575, 0.057],
        [0.82, 2.570, 0.058], [0.83, 2.566, 0.059], [0.84, 2.562, 0.059], [0.85, 2.559, 0.059],
        [0.86, 2.555, 0.058], [0.87, 2.552, 0.058], [0.88, 2.549, 0.057], [0.89, 2.547, 0.056],
        [0.90, 2.544, 0.054], [0.91, 2.541, 0.053], [0.92, 2.537, 0.053], [0.93, 2.535, 0.052],
        [0.94, 2.533, 0.051], [0.95, 2.531, 0.051], [0.96, 2.529, 0.049], [0.97, 2.527, 0.048],
        [0.98, 2.525, 0.046], [0.99, 2.522, 0.045], [1.00, 2.520, 0.043], [1.01, 2.517, 0.042],
        [1.02, 2.515, 0.042], [1.03, 2.512, 0.040], [1.04, 2.510, 0.039], [1.05, 2.507, 0.039],
        [1.06, 2.504, 0.038], [1.07, 2.502, 0.038], [1.08, 2.500, 0.037], [1.09, 2.498, 0.037],
        [1.10, 2.496, 0.036],
    ]
)


def n_fe2o3_o(l_um):
    return np.interp(l_um, _FE2O3_O_DATA[:, 0], _FE2O3_O_DATA[:, 1]) + 1j * np.interp(l_um, _FE2O3_O_DATA[:, 0], _FE2O3_O_DATA[:, 2])


def n_fe2o3_e(l_um):
    return np.interp(l_um, _FE2O3_E_DATA[:, 0], _FE2O3_E_DATA[:, 1]) + 1j * np.interp(l_um, _FE2O3_E_DATA[:, 0], _FE2O3_E_DATA[:, 2])


def n_ps(l_um):
    return np.sqrt(2.3809 + 0.01233 / (l_um**2 - 0.01615))


def n_sio2(l_um):
    l2 = l_um**2
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043**2) + 0.4079426 * l2 / (l2 - 0.1162414**2) + 0.8974794 * l2 / (l2 - 9.896161**2))


def n_pdms(l_um):
    return 1.3997 + 4.20e-3 / l_um**2


MATERIALS = {
    "TiO2-anatase": n_tio2_anatase,
    "Fe2O3-o": n_fe2o3_o,
    "Fe2O3-e": n_fe2o3_e,
    "PS": n_ps,
    "SiO2": n_sio2,
    "PDMS": n_pdms,
}


def resolve_material_model(name):
    if not isinstance(name, str) and np.isscalar(name):
        return lambda _l_um, value=float(np.real(name)): value
    if callable(name):
        return name
    if isinstance(name, str) and name not in MATERIALS:
        try:
            value = float(name)
            return lambda _l_um, numeric=value: numeric
        except ValueError:
            pass
    if name not in MATERIALS:
        raise KeyError(f"Unknown material: {name}")
    return MATERIALS[name]


def source_spectrum_lambda(lambda0_nm=855.0, fwhm_nm=56.0, npts=201):
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    lam = np.linspace(lambda0_nm - 5 * sigma, lambda0_nm + 5 * sigma, npts)
    return lam, np.exp(-0.5 * ((lam - lambda0_nm) / sigma) ** 2)


def trapezoid_weights(axis):
    axis = np.asarray(axis, dtype=float)
    w = np.zeros_like(axis)
    w[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    w[0] = 0.5 * (axis[1] - axis[0])
    w[-1] = 0.5 * (axis[-1] - axis[-2])
    return w


def normalize_intensity(values):
    values = np.asarray(values, dtype=float)
    vmax = float(np.max(values)) if values.size else 1.0
    return values / vmax if vmax > 0 else values


def interp_crossing(x, y, level, left=True):
    idx = np.where(y < level)[0]
    if len(idx) == 0:
        return float(x[0] if left else x[-1])
    if left:
        pos = idx[-1]
        x1, x2, y1, y2 = x[pos], x[pos + 1], y[pos], y[pos + 1]
    else:
        pos = idx[0]
        x1, x2, y1, y2 = x[pos - 1], x[pos], y[pos - 1], y[pos]
    return float(x1 + (level - y1) * (x2 - x1) / (y2 - y1 + 1e-30))


def axial_metrics(z_um, env):
    z_um = np.asarray(z_um, dtype=float)
    env = normalize_intensity(env)
    pk = int(np.argmax(env))
    peak_z = float(z_um[pk])
    zl = interp_crossing(z_um[: pk + 1], env[: pk + 1], 0.5, left=True)
    zr = interp_crossing(z_um[pk:], env[pk:], 0.5, left=False)
    fwhm = zr - zl
    mask = (z_um >= peak_z - 1.5 * fwhm) & (z_um <= peak_z + 1.5 * fwhm)
    centroid = float(np.sum(z_um[mask] * env[mask]) / (np.sum(env[mask]) + 1e-30))
    sidelobes = env.copy()
    sidelobes[mask] = 0.0
    peaks, _ = find_peaks(sidelobes, prominence=1e-6)
    psr = float(20 * np.log10(np.max(sidelobes[peaks]) + 1e-30)) if len(peaks) else float("-inf")
    return {"peak_z_um": peak_z, "centroid_z_um": centroid, "fwhm_um": float(fwhm), "psr_db": psr, "peak_value": float(env[pk])}


def gaussian_lateral_intensity(x_um, lambda0_nm, na, n_medium):
    if na <= 0:
        return np.ones_like(x_um, dtype=float)
    fwhm_um = 0.37 * (lambda0_nm / 1000.0) / (float(np.real(n_medium)) * na)
    return normalize_intensity(np.exp(-4 * np.log(2) * (x_um / max(fwhm_um, 1e-9)) ** 2))


DEFAULT_LIB_CANDIDATES = [
    Path(r"C:\Users\1\OneDrive - fzu.edu.cn (1)\Attachments\L_PSF_Work\libpytmatrix.dll"),
    Path(r"C:\Users\1\OneDrive - fzu.edu.cn (1)\Attachments\L_PSF_Work\libpytmatrix.so"),
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PYTMATRIX_SOURCE_ROOTS = [
    PROJECT_ROOT / "vendor" / "pytmatrix-0.3.3",
    PROJECT_ROOT / "vendor" / "pytmatrix-src" / "pytmatrix-0.3.3",
]
_TMATRIX_LIB = None
_CALCTMAT = None
_CALCAMPL = None
_TMATRIX_LIB_PATH = None
_TMATRIX_BACKEND = None
_PYTMATRIX_MODULE = None


@dataclass
class SourceConfig:
    lambda0_nm: float = 855.0
    fwhm_nm: float = 56.0
    n_lambda: int = 201


@dataclass
class GridConfig:
    z_span_um: float = 40.0
    n_z: int = 2001
    x_span_um: float = 8.0
    n_x: int = 129
    na: float = 0.05
    n_bfp_dense: int = 129
    n_bfp_sparse: int = 11
    bfp_extent_um: float = 1.0


@dataclass
class SolverConfig:
    mode: str = "low_na"
    particle_material: str = "TiO2-anatase"
    medium_material: str = "PDMS"
    diameter_nm: float = 200.0
    eps: float = 0.0
    beta_deg: float = 0.0
    amp_component: str = "S22"
    ideal: bool = False
    force_tmatrix: bool = False
    library_path: str | None = None


def _candidate_library_paths(library_path=None):
    candidates = []
    if library_path:
        candidates.append(Path(library_path))
    env_path = os.environ.get("PYTMATRIX_LIB")
    if env_path:
        candidates.append(Path(env_path))
    script_dir = Path(__file__).resolve().parent
    candidates.extend([script_dir / "libpytmatrix.dll", script_dir / "libpytmatrix.so", script_dir / "libpytmatrix.dylib"])
    candidates.extend(DEFAULT_LIB_CANDIDATES)
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def ensure_tmatrix_loaded(library_path=None):
    global _TMATRIX_LIB, _CALCTMAT, _CALCAMPL, _TMATRIX_LIB_PATH, _TMATRIX_BACKEND, _PYTMATRIX_MODULE
    if _TMATRIX_BACKEND is not None:
        return _TMATRIX_LIB_PATH
    errors = []
    for candidate in _candidate_library_paths(library_path):
        try:
            lib = ctypes.CDLL(str(candidate))
            _TMATRIX_LIB = lib
            _CALCTMAT = lib.calctmat_
            _CALCAMPL = lib.calcampl_
            _TMATRIX_LIB_PATH = str(candidate)
            _TMATRIX_BACKEND = "ctypes"
            return _TMATRIX_LIB_PATH
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    for source_root in LOCAL_PYTMATRIX_SOURCE_ROOTS:
        if source_root.exists() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    try:
        from pytmatrix.fortran_tm import pytmatrix as pytmatrix_module

        _PYTMATRIX_MODULE = pytmatrix_module
        _TMATRIX_LIB_PATH = "python:pytmatrix.fortran_tm.pytmatrix"
        _TMATRIX_BACKEND = "python"
        return _TMATRIX_LIB_PATH
    except Exception as error:  # pragma: no cover - diagnostic path
        errors.append(f"python:pytmatrix.fortran_tm.pytmatrix: {error}")
    raise FileNotFoundError("Unable to load libpytmatrix. " + " | ".join(errors))


def calc_sz(radius_um, wavelength_medium_um, m_rel, axis_ratio, *, thet0=90.0, thet=90.0, phi0=0.0, phi=180.0, alpha=0.0, beta=0.0, shape=-1, rat=1.0, ddelt=1e-3, ndgs=2, library_path=None):
    ensure_tmatrix_loaded(library_path=library_path)
    if _TMATRIX_BACKEND == "python":
        nmax = _PYTMATRIX_MODULE.calctmat(radius_um, rat, wavelength_medium_um, float(np.real(m_rel)), float(np.imag(m_rel)), axis_ratio, shape, ddelt, ndgs)
        s, z = _PYTMATRIX_MODULE.calcampl(nmax, wavelength_medium_um, thet0, thet, phi0, phi, alpha, beta)
        return np.asarray(s, dtype=np.complex128), np.asarray(z, dtype=np.float64)
    nmax = ctypes.c_int()
    args1 = [
        ctypes.c_double(radius_um),
        ctypes.c_double(rat),
        ctypes.c_double(wavelength_medium_um),
        ctypes.c_double(float(np.real(m_rel))),
        ctypes.c_double(float(np.imag(m_rel))),
        ctypes.c_double(axis_ratio),
        ctypes.c_int(shape),
        ctypes.c_double(ddelt),
        ctypes.c_int(ndgs),
        nmax,
    ]
    _CALCTMAT(*[ctypes.byref(x) for x in args1])
    s = np.zeros((2, 2), dtype=np.complex128, order="F")
    z = np.zeros((4, 4), dtype=np.float64, order="F")
    args2 = [ctypes.c_int(nmax.value), ctypes.c_double(wavelength_medium_um), ctypes.c_double(thet0), ctypes.c_double(thet), ctypes.c_double(phi0), ctypes.c_double(phi), ctypes.c_double(alpha), ctypes.c_double(beta)]
    _CALCAMPL(
        *[ctypes.byref(x) for x in args2],
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return s, z


def select_amplitude_component(s_matrix, amp_component="S22"):
    component = amp_component.upper()
    mapping = {
        "S11": s_matrix[0, 0],
        "S12": s_matrix[0, 1],
        "S21": s_matrix[1, 0],
        "S22": s_matrix[1, 1],
        "CO_POL": 0.5 * (s_matrix[0, 0] + s_matrix[1, 1]),
        "AVG_DIAG": 0.5 * (s_matrix[0, 0] + s_matrix[1, 1]),
    }
    if component not in mapping:
        raise ValueError(f"Unsupported amp_component: {amp_component}")
    return mapping[component]


def spectral_cube_to_xz(lambda_nm, spectral_cube, z_um, medium_material):
    cube = np.asarray(spectral_cube, dtype=np.complex128)
    cube = cube[:, None] if cube.ndim == 1 else cube
    lam_um = np.asarray(lambda_nm, dtype=float) / 1000.0
    medium_fn = resolve_material_model(medium_material)
    n_medium = np.array([float(np.real(medium_fn(l))) for l in lam_um], dtype=float)
    nu = n_medium / lam_um
    order = np.argsort(nu)
    kernel = np.exp(1j * 2 * np.pi * nu[order, None] * z_um[None, :]) * trapezoid_weights(nu[order])[:, None]
    return cube[order, :].T @ kernel


def mie_backscatter_spectrum(diameter_nm, particle_material, medium_material, lambda_nm):
    particle_fn = resolve_material_model(particle_material)
    medium_fn = resolve_material_model(medium_material)
    out = np.zeros(len(lambda_nm), dtype=np.complex128)
    for i, lam_nm in enumerate(lambda_nm):
        lam_um = lam_nm / 1000.0
        n_medium = medium_fn(lam_um)
        m = particle_fn(lam_um) / n_medium
        x = np.pi * diameter_nm * float(np.real(n_medium)) / lam_nm
        a, b = mie_ab(m, x)
        out[i] = s_back_full(a, b)
    return out


def tmatrix_backscatter_spectrum(diameter_nm, eps, beta_deg, particle_material, medium_material, lambda_nm, amp_component="S22", library_path=None):
    particle_fn = resolve_material_model(particle_material)
    medium_fn = resolve_material_model(medium_material)
    out = np.zeros(len(lambda_nm), dtype=np.complex128)
    radius_um = diameter_nm / 2000.0
    for i, lam_nm in enumerate(lambda_nm):
        lam_um = lam_nm / 1000.0
        n_medium = medium_fn(lam_um)
        s_matrix, _ = calc_sz(radius_um, lam_um / float(np.real(n_medium)), particle_fn(lam_um) / n_medium, 1.0 + eps, beta=beta_deg, library_path=library_path)
        out[i] = select_amplitude_component(s_matrix, amp_component=amp_component)
    return out


def _spherical_to_cart(theta_deg, phi_deg):
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)], dtype=float)


def _backscatter_basis(thet0_deg=90.0, phi0_deg=0.0):
    incident = _spherical_to_cart(thet0_deg, phi0_deg)
    backscatter = -incident
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(reference, backscatter)) > 0.99:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent_u = np.cross(reference, backscatter)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(backscatter, tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    return backscatter, tangent_u, tangent_v


def build_bfp_angle_map(na=0.05, n_bfp=129, thet0_deg=90.0, phi0_deg=0.0):
    pupil_axis = np.linspace(-1.0, 1.0, n_bfp)
    u_pupil, v_pupil = np.meshgrid(pupil_axis, pupil_axis)
    valid_mask = (u_pupil**2 + v_pupil**2) <= 1.0
    tx = na * u_pupil
    ty = na * v_pupil
    tz = np.sqrt(np.clip(1.0 - tx**2 - ty**2, 0.0, None))
    backscatter, tangent_u, tangent_v = _backscatter_basis(thet0_deg=thet0_deg, phi0_deg=phi0_deg)
    directions = backscatter[None, None, :] * tz[..., None] + tangent_u[None, None, :] * tx[..., None] + tangent_v[None, None, :] * ty[..., None]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    theta_deg = np.rad2deg(np.arccos(np.clip(directions[..., 2], -1.0, 1.0)))
    phi_deg = np.rad2deg(np.arctan2(directions[..., 1], directions[..., 0])) % 360.0
    return {"pupil_axis": pupil_axis, "u_pupil": u_pupil, "v_pupil": v_pupil, "valid_mask": valid_mask, "theta_deg": theta_deg, "phi_deg": phi_deg}


def _interpolate_sparse_complex_grid(sparse_samples, sparse_axis, dense_axis, dense_valid_mask):
    order = 3 if len(sparse_axis) >= 4 else 1
    real_spline = RectBivariateSpline(sparse_axis, sparse_axis, np.ascontiguousarray(np.real(sparse_samples)), kx=order, ky=order)
    imag_spline = RectBivariateSpline(sparse_axis, sparse_axis, np.ascontiguousarray(np.imag(sparse_samples)), kx=order, ky=order)
    dense = real_spline(dense_axis, dense_axis) + 1j * imag_spline(dense_axis, dense_axis)
    dense[~dense_valid_mask] = 0.0
    return dense


def build_particle_bfp_field(diameter_nm, eps, beta_deg, particle_material, medium_material, lambda_nm, *, na=0.05, n_bfp_dense=129, n_bfp_sparse=11, amp_component="S22", library_path=None):
    particle_fn = resolve_material_model(particle_material)
    medium_fn = resolve_material_model(medium_material)
    dense_map = build_bfp_angle_map(na=na, n_bfp=n_bfp_dense)
    sparse_map = build_bfp_angle_map(na=na, n_bfp=n_bfp_sparse)
    field_dense = np.zeros((n_bfp_dense, n_bfp_dense, len(lambda_nm)), dtype=np.complex128)
    radius_um = diameter_nm / 2000.0
    for k, lam_nm in enumerate(lambda_nm):
        lam_um = lam_nm / 1000.0
        n_medium = medium_fn(lam_um)
        sparse_samples = np.zeros((n_bfp_sparse, n_bfp_sparse), dtype=np.complex128)
        for row in range(n_bfp_sparse):
            for col in range(n_bfp_sparse):
                if not sparse_map["valid_mask"][row, col]:
                    continue
                s_matrix, _ = calc_sz(radius_um, lam_um / float(np.real(n_medium)), particle_fn(lam_um) / n_medium, 1.0 + eps, thet=sparse_map["theta_deg"][row, col], phi=sparse_map["phi_deg"][row, col], beta=beta_deg, library_path=library_path)
                sparse_samples[row, col] = select_amplitude_component(s_matrix, amp_component=amp_component)
        field_dense[:, :, k] = _interpolate_sparse_complex_grid(sparse_samples, sparse_map["pupil_axis"], dense_map["pupil_axis"], dense_map["valid_mask"])
    return {"field_cube": field_dense, "pupil_axis": dense_map["pupil_axis"], "u_pupil": dense_map["u_pupil"], "valid_mask": dense_map["valid_mask"]}


def build_ideal_bfp_field(lambda_nm, *, na=0.05, n_bfp_dense=129):
    dense_map = build_bfp_angle_map(na=na, n_bfp=n_bfp_dense)
    field_dense = np.zeros((n_bfp_dense, n_bfp_dense, len(lambda_nm)), dtype=np.complex128)
    field_dense[dense_map["valid_mask"], :] = 1.0
    return {"field_cube": field_dense, "pupil_axis": dense_map["pupil_axis"], "u_pupil": dense_map["u_pupil"], "valid_mask": dense_map["valid_mask"]}


def pupil_field_to_lateral_line(bundle, lambda_nm, x_um, na, medium_material):
    medium_fn = resolve_material_model(medium_material)
    weights_1d = trapezoid_weights(bundle["pupil_axis"])
    weights_2d = np.outer(weights_1d, weights_1d)
    mask = bundle["valid_mask"]
    u_flat = bundle["u_pupil"][mask]
    w_flat = weights_2d[mask]
    field_line = np.zeros((len(lambda_nm), len(x_um)), dtype=np.complex128)
    for k, lam_nm in enumerate(lambda_nm):
        k_medium = 2 * np.pi * float(np.real(medium_fn(lam_nm / 1000.0))) / (lam_nm / 1000.0)
        phase = np.exp(1j * k_medium * na * np.outer(x_um, u_flat))
        field_line[k, :] = phase @ (w_flat * bundle["field_cube"][:, :, k][mask])
    return field_line


def solve_low_na_slice(source, grid, solver):
    lambda_nm, source_power = source_spectrum_lambda(source.lambda0_nm, source.fwhm_nm, source.n_lambda)
    x_um = np.linspace(-0.5 * grid.x_span_um, 0.5 * grid.x_span_um, grid.n_x)
    z_um = np.linspace(-grid.z_span_um, grid.z_span_um, grid.n_z)
    medium_fn = resolve_material_model(solver.medium_material)
    if solver.ideal:
        spectrum = np.ones_like(lambda_nm, dtype=np.complex128)
        tmatrix_used = False
    elif solver.force_tmatrix or abs(solver.eps) > 0:
        spectrum = tmatrix_backscatter_spectrum(solver.diameter_nm, solver.eps, solver.beta_deg, solver.particle_material, solver.medium_material, lambda_nm, amp_component=solver.amp_component, library_path=solver.library_path)
        tmatrix_used = True
    else:
        spectrum = mie_backscatter_spectrum(solver.diameter_nm, solver.particle_material, solver.medium_material, lambda_nm)
        tmatrix_used = False
    field_xz = spectral_cube_to_xz(lambda_nm, source_power[:, None] * spectrum[:, None], z_um, solver.medium_material)
    axial_env = normalize_intensity(np.abs(field_xz[0, :]))
    lateral_env = gaussian_lateral_intensity(x_um, source.lambda0_nm, grid.na, medium_fn(source.lambda0_nm / 1000.0))
    intensity_xz = normalize_intensity(lateral_env[:, None] * axial_env[None, :])
    return {
        "mode": "low_na",
        "x_um": x_um,
        "z_um": z_um,
        "lambda_nm": lambda_nm,
        "intensity_xz": intensity_xz,
        "axial_env": axial_env,
        "metrics": axial_metrics(z_um, axial_env),
        "tmatrix_used": tmatrix_used,
        "tmatrix_library": _TMATRIX_LIB_PATH if tmatrix_used else None,
    }


def solve_full_na_slice(source, grid, solver):
    lambda_nm, source_power = source_spectrum_lambda(source.lambda0_nm, source.fwhm_nm, source.n_lambda)
    x_um = np.linspace(-0.5 * grid.x_span_um, 0.5 * grid.x_span_um, grid.n_x)
    z_um = np.linspace(-grid.z_span_um, grid.z_span_um, grid.n_z)
    if solver.ideal:
        bundle = build_ideal_bfp_field(lambda_nm, na=grid.na, n_bfp_dense=grid.n_bfp_dense)
        tmatrix_used = False
    else:
        bundle = build_particle_bfp_field(solver.diameter_nm, solver.eps, solver.beta_deg, solver.particle_material, solver.medium_material, lambda_nm, na=grid.na, n_bfp_dense=grid.n_bfp_dense, n_bfp_sparse=grid.n_bfp_sparse, amp_component=solver.amp_component, library_path=solver.library_path)
        tmatrix_used = True
    lateral_field = pupil_field_to_lateral_line(bundle, lambda_nm, x_um, grid.na, solver.medium_material)
    field_xz = spectral_cube_to_xz(lambda_nm, source_power[:, None] * lateral_field, z_um, solver.medium_material)
    intensity_xz = normalize_intensity(np.abs(field_xz))
    center_idx = int(np.argmin(np.abs(x_um)))
    return {
        "mode": "full_na",
        "x_um": x_um,
        "z_um": z_um,
        "lambda_nm": lambda_nm,
        "intensity_xz": intensity_xz,
        "axial_env": normalize_intensity(intensity_xz[center_idx, :]),
        "metrics": axial_metrics(z_um, intensity_xz[center_idx, :]),
        "tmatrix_used": tmatrix_used,
        "tmatrix_library": _TMATRIX_LIB_PATH if tmatrix_used else None,
        "pupil_shape": list(bundle["field_cube"].shape),
    }


def solve_oct_particle_response(source=None, grid=None, solver=None):
    source = source or SourceConfig()
    grid = grid or GridConfig()
    solver = solver or SolverConfig()
    if solver.mode == "low_na":
        result = solve_low_na_slice(source, grid, solver)
    elif solver.mode == "full_na":
        result = solve_full_na_slice(source, grid, solver)
    else:
        raise ValueError(f"Unsupported mode: {solver.mode}")
    result["source"] = asdict(source)
    result["grid"] = asdict(grid)
    result["solver"] = asdict(solver)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Non-spherical OCT low-NA / full-NA solver.")
    parser.add_argument("--mode", default="low_na", choices=["low_na", "full_na"])
    parser.add_argument("--ideal", action="store_true")
    parser.add_argument("--force-tmatrix", action="store_true")
    parser.add_argument("--particle-material", default="TiO2-anatase")
    parser.add_argument("--medium-material", default="PDMS")
    parser.add_argument("--diameter-nm", type=float, default=200.0)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--beta-deg", type=float, default=0.0)
    parser.add_argument("--amp-component", default="S22", choices=["S11", "S12", "S21", "S22", "CO_POL", "AVG_DIAG"])
    parser.add_argument("--lambda0-nm", type=float, default=855.0)
    parser.add_argument("--fwhm-nm", type=float, default=56.0)
    parser.add_argument("--n-lambda", type=int, default=201)
    parser.add_argument("--z-span-um", type=float, default=40.0)
    parser.add_argument("--n-z", type=int, default=2001)
    parser.add_argument("--x-span-um", type=float, default=8.0)
    parser.add_argument("--n-x", type=int, default=129)
    parser.add_argument("--na", type=float, default=0.05)
    parser.add_argument("--n-bfp-dense", type=int, default=129)
    parser.add_argument("--n-bfp-sparse", type=int, default=11)
    parser.add_argument("--lib-path")
    parser.add_argument("--output-json")
    parser.add_argument("--output-npz")
    return parser


def main():
    args = build_arg_parser().parse_args()
    result = solve_oct_particle_response(
        SourceConfig(args.lambda0_nm, args.fwhm_nm, args.n_lambda),
        GridConfig(args.z_span_um, args.n_z, args.x_span_um, args.n_x, args.na, args.n_bfp_dense, args.n_bfp_sparse),
        SolverConfig(args.mode, args.particle_material, args.medium_material, args.diameter_nm, args.eps, args.beta_deg, args.amp_component, args.ideal, args.force_tmatrix, args.lib_path),
    )
    if args.output_npz:
        np.savez_compressed(args.output_npz, x_um=result["x_um"], z_um=result["z_um"], lambda_nm=result["lambda_nm"], intensity_xz=result["intensity_xz"], axial_env=result["axial_env"])
    summary = {
        "mode": result["mode"],
        "metrics": result["metrics"],
        "tmatrix_used": result["tmatrix_used"],
        "tmatrix_library": result["tmatrix_library"],
        "source": result["source"],
        "grid": result["grid"],
        "solver": result["solver"],
    }
    if "pupil_shape" in result:
        summary["pupil_shape"] = result["pupil_shape"]
    payload = json.dumps(summary, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
