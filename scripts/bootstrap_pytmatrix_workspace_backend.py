import argparse
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / "tmp" / "pytmatrix_pkg"
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "pytmatrix-0.3.3"
SHIM_ROOT = PROJECT_ROOT / "tmp" / "python_shims"
ZIP_URL = "https://github.com/jleinonen/pytmatrix/releases/download/0.3.3/pytmatrix-0.3.3.zip"
PYF_URL = "https://raw.githubusercontent.com/jleinonen/pytmatrix/master/pytmatrix/fortran_tm/pytmatrix.pyf"
ZIP_PATH = TMP_ROOT / "pytmatrix-0.3.3.zip"
PYF_RELATIVE = Path("pytmatrix") / "fortran_tm" / "pytmatrix.pyf"
DEFAULT_BUILD_PYTHON = Path(r"C:\Users\1\anaconda3\envs\oct_psf\python.exe")


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def extract_zip(archive: Path, dest: Path):
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest.parent)


def ensure_source_tree():
    actions = []
    if not ZIP_PATH.exists():
        download(ZIP_URL, ZIP_PATH)
        actions.append(f"downloaded_zip:{ZIP_PATH}")
    extract_zip(ZIP_PATH, VENDOR_ROOT)
    actions.append(f"extracted_zip:{VENDOR_ROOT}")
    pyf_path = VENDOR_ROOT / PYF_RELATIVE
    if not pyf_path.exists():
        download(PYF_URL, pyf_path)
        actions.append(f"downloaded_pyf:{pyf_path}")
    return actions


def ensure_python_shim():
    SHIM_ROOT.mkdir(parents=True, exist_ok=True)
    shim_file = SHIM_ROOT / "sitecustomize.py"
    shim_file.write_text(
        "import importlib\n"
        "import re\n"
        "import sys, types\n"
        "module = types.ModuleType('distutils.msvccompiler')\n"
        "def get_build_version():\n"
        "    return None\n"
        "module.get_build_version = get_build_version\n"
        "try:\n"
        "    import setuptools._distutils._msvccompiler as _msvc\n"
        "    for name in dir(_msvc):\n"
        "        if not name.startswith('__'):\n"
        "            setattr(module, name, getattr(_msvc, name))\n"
        "except Exception:\n"
        "    pass\n"
        "sys.modules.setdefault('distutils.msvccompiler', module)\n"
        "def _patched_get_msvcr(orig):\n"
        "    def inner():\n"
        "        match = re.search(r'MSC v\\\\.(\\\\d{4})', sys.version)\n"
        "        if match:\n"
        "            msc_ver = int(match.group(1))\n"
        "            if 1900 <= msc_ver < 2000:\n"
        "                return ['vcruntime140']\n"
        "        return orig()\n"
        "    return inner\n"
        "for modname in ('distutils.cygwinccompiler', 'setuptools._distutils.cygwinccompiler'):\n"
        "    try:\n"
        "        mod = importlib.import_module(modname)\n"
        "        if hasattr(mod, 'get_msvcr'):\n"
        "            mod.get_msvcr = _patched_get_msvcr(mod.get_msvcr)\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    return shim_file


def attempt_build(build_python: Path, compiler: str | None = None):
    result = {
        "python": str(build_python),
        "exists": build_python.exists(),
    }
    if not build_python.exists():
        result["attempted"] = False
        result["reason"] = "build_python_missing"
        return result
    command = [str(build_python), "setup.py", "build_ext"]
    if compiler:
        command.extend(["--compiler", compiler])
    command.append("--inplace")
    env = dict(os.environ)
    shim_file = ensure_python_shim()
    env["PYTHONPATH"] = str(SHIM_ROOT) + (";" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(command, cwd=str(VENDOR_ROOT), capture_output=True, text=True, timeout=300, env=env)
    result["attempted"] = True
    result["shim"] = str(shim_file)
    result["command"] = command
    result["returncode"] = completed.returncode
    result["stdout_tail"] = "\n".join(completed.stdout.splitlines()[-40:])
    result["stderr_tail"] = "\n".join(completed.stderr.splitlines()[-40:])
    built = list((VENDOR_ROOT / "pytmatrix" / "fortran_tm").glob("pytmatrix*.pyd")) + list((VENDOR_ROOT / "pytmatrix" / "fortran_tm").glob("pytmatrix*.so"))
    result["built_extensions"] = [str(path) for path in built]
    return result


def bootstrap(build_python: Path = DEFAULT_BUILD_PYTHON, compiler: str | None = None):
    report = {
        "vendor_root": str(VENDOR_ROOT),
        "actions": ensure_source_tree(),
        "build": attempt_build(build_python, compiler=compiler),
    }
    report["ready_for_import"] = bool(report["build"].get("built_extensions"))
    if report["ready_for_import"]:
        report["next_step"] = "Run the solver normally; it will auto-discover the vendored pytmatrix backend."
    else:
        report["next_step"] = "Source tree is configured in the workspace, but the compiled backend is still missing."
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and optionally build a workspace-local pytmatrix backend.")
    parser.add_argument(
        "--build-python",
        default=str(DEFAULT_BUILD_PYTHON),
        help="Python interpreter used to run setup.py build_ext --inplace.",
    )
    parser.add_argument(
        "--compiler",
        default=None,
        help="Optional distutils compiler name, for example 'mingw32'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            bootstrap(build_python=Path(args.build_python), compiler=args.compiler),
            indent=2,
            ensure_ascii=False,
        )
    )
