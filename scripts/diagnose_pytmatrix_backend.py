import ctypes
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tmp" / "pytmatrix_pkg"
ARCHIVE_PATH = TMP_ROOT / "pytmatrix-0.3.3.tar.gz"

PYTHON_CANDIDATES = [
    Path(sys.executable),
    Path(r"C:\Users\1\anaconda3\python.exe"),
    Path(r"C:\Users\1\anaconda3\envs\oct_psf\python.exe"),
    Path(r"C:\ProgramData\anaconda3\python.exe"),
    Path(r"C:\ProgramData\miniconda3\python.exe"),
]

LIBRARY_CANDIDATES = [
    Path(__file__).resolve().parent / "libpytmatrix.dll",
    Path(__file__).resolve().parent / "libpytmatrix.so",
    Path(__file__).resolve().parent / "libpytmatrix.dylib",
    Path(r"C:\Users\1\OneDrive - fzu.edu.cn (1)\Attachments\L_PSF_Work\libpytmatrix.dll"),
    Path(r"C:\Users\1\OneDrive - fzu.edu.cn (1)\Attachments\L_PSF_Work\libpytmatrix.so"),
]

COMPILER_CANDIDATES = {
    "gfortran": [
        Path(r"C:\Users\1\anaconda3\Library\mingw-w64\bin\gfortran.exe"),
        Path(r"C:\Users\1\anaconda3\envs\oct_psf\Library\mingw-w64\bin\gfortran.exe"),
        Path(r"C:\ProgramData\anaconda3\Library\mingw-w64\bin\gfortran.exe"),
    ],
    "gcc": [
        Path(r"C:\Users\1\anaconda3\Library\mingw-w64\bin\gcc.exe"),
        Path(r"C:\Users\1\anaconda3\envs\oct_psf\Library\mingw-w64\bin\gcc.exe"),
        Path(r"C:\ProgramData\anaconda3\Library\mingw-w64\bin\gcc.exe"),
    ],
    "cl": [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"),
    ],
}


def unique_paths(paths):
    seen = set()
    out = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def probe_workspace_writability():
    probe_dir = TMP_ROOT
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_file = probe_dir / "write_probe.txt"
    try:
        probe_file.write_text("ok\n", encoding="utf-8")
        probe_file.unlink()
        return {"path": str(probe_dir), "writable": True}
    except Exception as error:  # pragma: no cover - platform-specific
        return {"path": str(probe_dir), "writable": False, "reason": str(error)}


def probe_python(executable: Path):
    result = {"path": str(executable), "exists": executable.exists()}
    if not executable.exists():
        return result
    snippet = r"""
import importlib.util, json, sys
mods = ["numpy", "scipy", "numpy.distutils", "pytmatrix", "pytmatrix.fortran_tm.pytmatrix"]
payload = {"executable": sys.executable, "version": sys.version.split()[0]}
for mod in mods:
    try:
        spec = importlib.util.find_spec(mod)
        payload[mod + "_available"] = spec is not None
        payload[mod + "_origin"] = None if spec is None else getattr(spec, "origin", None)
    except Exception as exc:
        payload[mod + "_available"] = False
        payload[mod + "_origin"] = None
        payload[mod + "_error"] = str(exc)
print(json.dumps(payload))
"""
    completed = subprocess.run([str(executable), "-c", snippet], capture_output=True, text=True, timeout=30)
    result["returncode"] = completed.returncode
    result["stderr"] = completed.stderr.strip()
    if completed.returncode != 0:
        result["stdout"] = completed.stdout.strip()
        return result
    try:
        result["probe"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result["stdout"] = completed.stdout.strip()
    return result


def probe_library(path: Path):
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        ctypes.CDLL(str(path))
        result["loadable"] = True
    except OSError as error:
        result["loadable"] = False
        result["reason"] = str(error)
    return result


def probe_compilers():
    report = {}
    for tool, candidates in COMPILER_CANDIDATES.items():
        info = {
            "on_path": shutil.which(tool),
            "existing_candidates": [str(path) for path in candidates if path.exists()],
        }
        report[tool] = info
    return report


def probe_archive(path: Path):
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    result["size_bytes"] = path.stat().st_size
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        result["contains_setup_py"] = "pytmatrix-0.3.3/setup.py" in names
        result["contains_pyf"] = "pytmatrix-0.3.3/pytmatrix/fortran_tm/pytmatrix.pyf" in names
        result["contains_fortran_sources"] = [
            name
            for name in names
            if name.startswith("pytmatrix-0.3.3/pytmatrix/fortran_tm/") and name.endswith((".f", ".f90", ".pyf"))
        ]
        if result["contains_setup_py"]:
            setup_text = archive.extractfile("pytmatrix-0.3.3/setup.py").read().decode("utf-8", errors="replace")
            result["uses_numpy_distutils"] = "numpy.distutils" in setup_text
            result["defines_fortran_extension"] = "config.add_extension('fortran_tm.pytmatrix'" in setup_text
    return result


def build_recommendations(report):
    recommendations = []
    loadable_libs = [item for item in report["libraries"] if item.get("loadable")]
    python_backends = [
        item for item in report["python_envs"]
        if item.get("probe", {}).get("pytmatrix.fortran_tm.pytmatrix_available")
    ]
    has_gfortran = bool(report["compilers"]["gfortran"]["on_path"] or report["compilers"]["gfortran"]["existing_candidates"])
    has_gcc = bool(report["compilers"]["gcc"]["on_path"] or report["compilers"]["gcc"]["existing_candidates"])
    has_msvc = bool(report["compilers"]["cl"]["on_path"] or report["compilers"]["cl"]["existing_candidates"])
    archive_exists = report["archive"].get("exists")
    archive_complete = archive_exists and report["archive"].get("contains_setup_py") and report["archive"].get("contains_pyf")

    if loadable_libs or python_backends:
        verdict = "backend_available"
        recommendations.append("Set `--lib-path` to the loadable library, or run the solver under the Python environment that already imports `pytmatrix`.")
        return verdict, recommendations

    if archive_exists and not archive_complete:
        verdict = "source_archive_incomplete"
        recommendations.append("The local `pytmatrix-0.3.3` source archive is incomplete for a local build because `pytmatrix/fortran_tm/pytmatrix.pyf` is missing.")
        recommendations.append("Fetch the full upstream source tree or release asset before attempting any local compilation.")
        return verdict, recommendations

    if archive_complete and (has_gfortran or has_gcc or has_msvc):
        verdict = "source_available_but_unbuilt"
        recommendations.append("Build `pytmatrix 0.3.3` from the local source archive using a Python environment with `numpy<2` and `numpy.distutils` available.")
        recommendations.append("Prefer a dedicated Python 3.10/3.11 environment because upstream `pytmatrix` 0.3.3 uses `numpy.distutils` in `setup.py`.")
        return verdict, recommendations

    if archive_complete:
        verdict = "missing_compiler_toolchain"
        recommendations.append("Install a Fortran-capable toolchain first. `pytmatrix` 0.3.3 exposes a Fortran extension and cannot be built here without `gfortran` or an equivalent supported compiler.")
        recommendations.append("After the compiler exists, build `pytmatrix` in a Python 3.10/3.11 environment and point the solver at the resulting backend.")
        return verdict, recommendations

    verdict = "missing_backend_and_source"
    recommendations.append("Keep the current solver on Mie/ideal mode until a `pytmatrix` backend or `libpytmatrix` library is provisioned.")
    recommendations.append("Re-download the `pytmatrix 0.3.3` source archive into the project tmp directory before attempting a local build.")
    return verdict, recommendations


def diagnose():
    report = {
        "workspace_writable": probe_workspace_writability(),
        "python_envs": [probe_python(path) for path in unique_paths(PYTHON_CANDIDATES)],
        "libraries": [probe_library(path) for path in unique_paths(LIBRARY_CANDIDATES)],
        "compilers": probe_compilers(),
        "archive": probe_archive(ARCHIVE_PATH),
    }
    verdict, recommendations = build_recommendations(report)
    report["verdict"] = verdict
    report["recommended_next_steps"] = recommendations
    return report


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2, ensure_ascii=False))
