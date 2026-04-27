from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from subprocess import Popen

import uvicorn


def _ensure_frontend_dependencies(frontend_dir_arg: Path) -> None:
    node_modules_dir = frontend_dir_arg / "node_modules"
    vite_bin = node_modules_dir / ".bin" / "vite"
    if node_modules_dir.exists() and vite_bin.exists():
        return

    lock_exists = (frontend_dir_arg / "package-lock.json").exists()
    install_cmd = ["npm", "ci"] if lock_exists else ["npm", "install"]
    subprocess.run(install_cmd, cwd=frontend_dir_arg, check=True)


def _start_frontend(frontend_dir_arg: Path) -> Popen[str]:
    if not frontend_dir_arg.exists():
        raise FileNotFoundError(f"frontend directory not found: {frontend_dir_arg}")

    _ensure_frontend_dependencies(frontend_dir_arg)
    env = os.environ.copy()
    cmd = ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
    return subprocess.Popen(cmd, cwd=frontend_dir_arg, env=env, text=True)


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Gradulate backend and frontend dev servers.")
    parser.add_argument("--backend-only", action="store_true", help="Start only FastAPI backend.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    project_root = Path(__file__).resolve().parent.parent
    frontend_dir = project_root / "ui" / "frontend"
    frontend_process: subprocess.Popen[str] | None = None

    if not args.backend_only:
        try:
            frontend_process = _start_frontend(frontend_dir)
            print("Frontend started at http://127.0.0.1:5173")
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            print("Fallback to backend only mode.", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - runtime environment differences
            print(f"failed to start frontend: {exc}", file=sys.stderr)
            print("Fallback to backend only mode.", file=sys.stderr)

    try:
        print("Backend started at http://127.0.0.1:8000")
        uvicorn.run("ui.main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        pass
    finally:
        _terminate_process(frontend_process)
