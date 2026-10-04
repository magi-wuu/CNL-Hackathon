#!/usr/bin/env python3
"""Start the local dashboard from any working directory, without shell setup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
RUNTIME = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_node():
    candidates = [shutil.which("node"), str(RUNTIME / "node/bin/node")]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            version = subprocess.check_output([candidate, "--version"], text=True, timeout=5).strip()
            if int(version.lstrip("v").split(".")[0]) >= 22:
                return Path(candidate).resolve(), version
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    raise RuntimeError("Node.js 22 or newer is required. Install Node.js, then run this launcher again.")


def process_environment(node):
    env = os.environ.copy()
    env["PATH"] = str(node.parent) + os.pathsep + env.get("PATH", "")
    return env


def pnpm_command(node):
    installed = shutil.which("pnpm")
    if installed:
        return [installed]
    bundled = RUNTIME / "node/node_modules/pnpm/bin/pnpm.mjs"
    if bundled.is_file():
        return [str(node), str(bundled)]
    # A regular Node install normally provides npx; it can run pnpm without a global install.
    npx = shutil.which("npx", path=process_environment(node)["PATH"])
    if npx:
        return [npx, "--yes", "pnpm@11.19.0"]
    raise RuntimeError("pnpm could not be found. Install pnpm 11, then run: python run.py --install")


def install(node):
    if not PYTHON.is_file():
        candidates = [shutil.which("python3.12"), str(RUNTIME / "python/bin/python3"), sys.executable]
        creator = None
        for candidate in candidates:
            if not candidate or not Path(candidate).is_file():
                continue
            version = subprocess.check_output([candidate, "-c", "import sys; print('.'.join(map(str,sys.version_info[:2])))"], text=True).strip()
            if version == "3.12":
                creator = candidate
                break
        if creator is None:
            raise RuntimeError("Install Python 3.12 to create the dashboard's environment.")
        subprocess.run([creator, "-m", "venv", str(ROOT / ".venv")], check=True)
    subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "backend/requirements.txt")], cwd=ROOT, check=True)
    subprocess.run([*pnpm_command(node), "install", "--frozen-lockfile"], cwd=FRONTEND, env=process_environment(node), check=True)
    print("Dependencies ready. Start with: python run.py", flush=True)


def occupied(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(.5)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def existing_service(kind, port):
    if not occupied(port):
        return False
    path = "/api/health" if kind == "backend" else "/"
    try:
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
            content = response.read(65536).decode("utf-8")
        if kind == "backend":
            payload = json.loads(content)
            valid = payload.get("status") == "ready" and payload.get("classes") == ["No leak", "LOCA", "LOCAC", "SGATR", "SGBTR", "SLBIC", "FLB", "LLB"]
        else:
            valid = "<title>Reactor Control Panel</title>" in content
        if valid:
            return True
    except (OSError, ValueError):
        pass
    raise RuntimeError(f"Port {port} is already in use by a service that could not be identified as this dashboard. Check the other terminal or wait for its startup to finish.")


def check_dependencies(node, version):
    if not PYTHON.is_file():
        raise RuntimeError("Dashboard Python environment is missing. Run: python run.py --install")
    subprocess.run([str(PYTHON), "-c", "import uvicorn, fastapi"], cwd=ROOT, check=True)
    if not (FRONTEND / "node_modules/vite/bin/vite.js").is_file():
        raise RuntimeError("Frontend dependencies are missing. Run: python run.py --install")
    print(f"Project: {ROOT}\nPython: {PYTHON}\nNode: {node} ({version})", flush=True)


def start(node, service):
    children = []
    try:
        for kind, port in [("backend", 8000), ("frontend", 5173)]:
            if service != "all" and service != kind:
                continue
            if existing_service(kind, port):
                print(f"{kind.capitalize()} already running at http://127.0.0.1:{port}", flush=True)
                continue
            if kind == "backend":
                command = [str(PYTHON), "-m", "uvicorn", "backend.app:app", "--app-dir", str(ROOT), "--host", "127.0.0.1", "--port", str(port)]
                cwd = ROOT
            else:
                command = [str(node), str(FRONTEND / "node_modules/vite/bin/vite.js"), "--host", "127.0.0.1", "--port", str(port), "--strictPort"]
                cwd = FRONTEND
            print(f"Starting {kind}…", flush=True)
            children.append((kind, subprocess.Popen(command, cwd=cwd, env=process_environment(node))))
        print("\nDashboard: http://127.0.0.1:5173/", flush=True)
        if children:
            print("Keep this terminal open. Ctrl+C stops the services started by this launcher.\n", flush=True)
        while children:
            for kind, child in children:
                code = child.poll()
                if code is not None:
                    raise RuntimeError(f"{kind.capitalize()} exited (code {code}). Check its message above.")
            time.sleep(.25)
    except KeyboardInterrupt:
        print("\nStopping services started by this launcher…", flush=True)
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Install the local Python and frontend dependencies")
    parser.add_argument("--check", action="store_true", help="Check tool paths and dependencies without starting servers")
    parser.add_argument("--service", choices=["all", "backend", "frontend"], default="all", help="Start both services or one service")
    args = parser.parse_args()
    try:
        node, version = find_node()
        if args.install:
            install(node)
        else:
            check_dependencies(node, version)
            if not args.check:
                start(node, args.service)
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"\nCannot start dashboard: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
