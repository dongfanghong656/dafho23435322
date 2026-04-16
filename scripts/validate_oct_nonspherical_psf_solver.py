import importlib.util
import json
from pathlib import Path

import numpy as np

from oct_nonspherical_psf_solver import GridConfig, SolverConfig, SourceConfig, ensure_tmatrix_loaded, solve_oct_particle_response


LEGACY_MILESTONE1 = Path(r"C:\Users\1\OneDrive - fzu.edu.cn (1)\Attachments\L_PSF_Work\run_tmatrix_oct_direct_milestone1.py")


def load_legacy_module():
    if not LEGACY_MILESTONE1.exists():
        return None
    spec = importlib.util.spec_from_file_location("legacy_tmatrix_m1", LEGACY_MILESTONE1)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate(lib_path=None):
    report = {"checks": []}
    ideal = solve_oct_particle_response(
        SourceConfig(n_lambda=121),
        GridConfig(z_span_um=20.0, n_z=801, x_span_um=4.0, n_x=41),
        SolverConfig(mode="low_na", medium_material=1.40, ideal=True),
    )
    center_idx = int(np.argmin(np.abs(ideal["x_um"])))
    report["checks"].append(
        {
            "name": "low_na_ideal",
            "passed": bool(np.max(ideal["intensity_xz"]) > 0.999 and abs(ideal["metrics"]["peak_z_um"]) < 0.15 and ideal["intensity_xz"][center_idx, ideal["intensity_xz"].shape[1] // 2] >= ideal["intensity_xz"][0, ideal["intensity_xz"].shape[1] // 2]),
            "metrics": ideal["metrics"],
        }
    )
    sphere = solve_oct_particle_response(
        SourceConfig(n_lambda=121),
        GridConfig(z_span_um=20.0, n_z=801, x_span_um=4.0, n_x=41),
        SolverConfig(mode="low_na", particle_material="TiO2-anatase", medium_material="PDMS", diameter_nm=150.0),
    )
    report["checks"].append(
        {
            "name": "low_na_mie_smoke",
            "passed": bool(np.isfinite(sphere["intensity_xz"]).all() and sphere["metrics"]["fwhm_um"] > 0 and not sphere["tmatrix_used"]),
            "metrics": sphere["metrics"],
        }
    )
    try:
        tmatrix_path = ensure_tmatrix_loaded(lib_path)
    except FileNotFoundError as error:
        report["checks"].append({"name": "tmatrix_available", "passed": False, "skipped": True, "reason": str(error)})
        return report
    legacy = load_legacy_module()
    if legacy is None:
        report["checks"].append({"name": "legacy_milestone1_available", "passed": False, "skipped": True, "reason": f"Missing {LEGACY_MILESTONE1}"})
        return report
    low_na_exact = solve_oct_particle_response(
        SourceConfig(lambda0_nm=855.0, fwhm_nm=56.0, n_lambda=801),
        GridConfig(z_span_um=20.0, n_z=8001, x_span_um=4.0, n_x=41),
        SolverConfig(mode="low_na", particle_material=2.48, medium_material=1.40, diameter_nm=300.0, force_tmatrix=True, library_path=tmatrix_path),
    )
    legacy_metrics, _, _ = legacy.run_case(300.0, 0.0, 0.0, library_path=tmatrix_path)
    compare = {
        "peak_delta_um": abs(low_na_exact["metrics"]["peak_z_um"] - legacy_metrics["peak_opd_um"]),
        "centroid_delta_um": abs(low_na_exact["metrics"]["centroid_z_um"] - legacy_metrics["centroid_opd_um"]),
        "fwhm_delta_um": abs(low_na_exact["metrics"]["fwhm_um"] - legacy_metrics["fwhm_opd_um"]),
        "psr_delta_db": abs(low_na_exact["metrics"]["psr_db"] - legacy_metrics["psr_db"]),
    }
    report["checks"].append(
        {
            "name": "low_na_vs_legacy_milestone1",
            "passed": bool(compare["peak_delta_um"] < 0.05 and compare["centroid_delta_um"] < 0.05 and compare["fwhm_delta_um"] < 0.05 and compare["psr_delta_db"] < 0.5),
            "deltas": compare,
            "metrics": low_na_exact["metrics"],
        }
    )
    full_na = solve_oct_particle_response(
        SourceConfig(n_lambda=41),
        GridConfig(z_span_um=8.0, n_z=301, x_span_um=3.0, n_x=41, na=0.05, n_bfp_dense=49, n_bfp_sparse=7),
        SolverConfig(mode="full_na", particle_material=2.48, medium_material=1.40, diameter_nm=200.0, eps=0.10, beta_deg=45.0, library_path=tmatrix_path),
    )
    peak_index = np.unravel_index(np.argmax(full_na["intensity_xz"]), full_na["intensity_xz"].shape)
    report["checks"].append(
        {
            "name": "full_na_tmatrix_smoke",
            "passed": bool(np.isfinite(full_na["intensity_xz"]).all() and abs(full_na["x_um"][peak_index[0]]) <= 0.25 and full_na["metrics"]["fwhm_um"] > 0),
            "metrics": full_na["metrics"],
            "peak_x_um": float(full_na["x_um"][peak_index[0]]),
            "pupil_shape": full_na.get("pupil_shape"),
        }
    )
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
