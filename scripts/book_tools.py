#!/usr/bin/env python3
"""Helper commands for the conference Makefile."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
LOCAL_JUPYTER = ROOT / ".jupyter-local"
LIVE_LOG = ROOT / ".jupyter-live.log"
DEFAULT_VENV = ".venv"
DEFAULT_NOTEBOOKS_DIR = "notebooks"


def bool_arg(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


def venv_dir() -> Path:
    configured = os.environ.get("VENV_DIR", DEFAULT_VENV)
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    return path


def venv_python() -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_dir() / scripts_dir / executable


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def configured_notebooks_dir(configured: str | None = None) -> Path:
    path = Path(configured or DEFAULT_NOTEBOOKS_DIR)
    if not path.is_absolute():
        path = ROOT / path
    return path


def project_relative_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        raise SystemExit(
            f"Notebook directory must be inside the project: {display_path(path)}"
        )


def project_env() -> dict[str, str]:
    env = os.environ.copy()

    local_dirs = {
        "JUPYTER_CONFIG_DIR": LOCAL_JUPYTER / "config",
        "JUPYTER_DATA_DIR": LOCAL_JUPYTER / "data",
        "JUPYTER_RUNTIME_DIR": LOCAL_JUPYTER / "runtime",
        "IPYTHONDIR": LOCAL_JUPYTER / "ipython",
        "MPLCONFIGDIR": LOCAL_JUPYTER / "matplotlib",
    }
    for key, path in local_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)

    pythonpath = [str(ROOT), str(CHAPTERS)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def run(command: Sequence[object], *, env: Optional[dict[str, str]] = None) -> None:
    subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        env=project_env() if env is None else env,
        check=True,
    )


def venv_has_module(module: str) -> bool:
    if not venv_python().exists():
        return False

    result = subprocess.run(
        [
            str(venv_python()),
            "-c",
            f"import importlib.util; raise SystemExit(importlib.util.find_spec({module!r}) is None)",
        ],
        cwd=ROOT,
        env=project_env(),
    )
    return result.returncode == 0


def ensure_venv() -> None:
    python = venv_python()
    if python.exists():
        return

    print(f"Creating local Python environment in {display_path(venv_dir())}")
    run([sys.executable, "-m", "venv", venv_dir()], env=os.environ.copy())


def install_requirements() -> None:
    requirements = ROOT / "requirements.txt"
    ensure_venv()
    print(f"Installing requirements into {display_path(venv_dir())}")
    run(
        [
            venv_python(),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            requirements,
        ],
        env=os.environ.copy(),
    )


def ensure_ready_for_notebooks() -> None:
    if venv_has_module("jupytext") and venv_has_module("jupyterlab"):
        return

    print("Local environment is missing notebook requirements; installing them now.")
    install_requirements()


def public_help(_args: argparse.Namespace) -> None:
    print(
        "Conference commands:\n"
        "  make install   Create .venv and install the Python requirements\n"
        "  make start     Generate notebooks in notebooks/ and open JupyterLab there\n"
        "\n"
        "The environment lives inside this folder and is not committed to git."
    )


def install(_args: argparse.Namespace) -> None:
    print(f"Bootstrap Python: {sys.executable}")
    install_requirements()
    print("\nInstallation finished. Start the notebooks with: make start")


def sync_static_assets(notebooks_dir: Path) -> None:
    source = CHAPTERS / "_static"
    if not source.exists():
        return

    destination = notebooks_dir / "_static"
    if source.resolve() == destination.resolve():
        return

    shutil.copytree(source, destination, dirs_exist_ok=True)


def remove_legacy_chapter_notebooks(generated_paths: Sequence[Path] = ()) -> int:
    generated = {path.resolve() for path in generated_paths}
    removed = 0
    for notebook in CHAPTERS.glob("*.ipynb"):
        if notebook.resolve() in generated:
            continue
        notebook.unlink()
        removed += 1
    return removed


def remove_legacy_chapter_checkpoints() -> bool:
    checkpoints = CHAPTERS / ".ipynb_checkpoints"
    if not checkpoints.exists():
        return False

    shutil.rmtree(checkpoints)
    return True


def notebooks(args: Optional[argparse.Namespace] = None) -> None:
    ensure_ready_for_notebooks()

    notebooks_dir = configured_notebooks_dir(
        getattr(args, "notebooks_dir", DEFAULT_NOTEBOOKS_DIR)
    )
    project_relative_path(notebooks_dir)
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(CHAPTERS.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No markdown notebooks found in {CHAPTERS}")

    generated_paths: list[Path] = []
    for md_file in md_files:
        notebook = notebooks_dir / md_file.with_suffix(".ipynb").name
        generated_paths.append(notebook)
        print(f"Writing {display_path(notebook)}", flush=True)
        run(
            [
                venv_python(),
                "-m",
                "jupytext",
                "--to",
                "ipynb",
                "--output",
                notebook,
                md_file,
            ]
        )

    sync_static_assets(notebooks_dir)
    removed = remove_legacy_chapter_notebooks(generated_paths)
    removed_checkpoints = remove_legacy_chapter_checkpoints()
    if removed:
        print(f"Removed {removed} legacy generated notebooks from {display_path(CHAPTERS)}/.")
    if removed_checkpoints:
        print(f"Removed legacy checkpoints from {display_path(CHAPTERS / '.ipynb_checkpoints')}/.")

    print(f"\nGenerated {len(md_files)} notebooks in {display_path(notebooks_dir)}/.")


def jupyter_server_command(args: argparse.Namespace) -> list[object]:
    return [
        venv_python(),
        "-m",
        "jupyter",
        "server",
        "--no-browser",
        f"--ServerApp.root_dir={ROOT}",
        f"--ServerApp.ip={args.host}",
        f"--ServerApp.port={args.port}",
        f"--IdentityProvider.token={args.token}",
        "--ServerApp.allow_origin=*",
    ]


def start(args: argparse.Namespace) -> None:
    ensure_ready_for_notebooks()
    notebooks(args)

    notebooks_dir = configured_notebooks_dir(args.notebooks_dir)
    opened_path = project_relative_path(notebooks_dir)
    url = (
        f"http://{args.host}:{args.port}/lab/tree/"
        f"{opened_path.as_posix()}?token={args.token}"
    )
    print("\nStarting JupyterLab with the generated notebooks folder.")
    print(f"Open: {url}")

    command: list[object] = [
        venv_python(),
        "-m",
        "jupyter",
        "lab",
        str(opened_path),
        f"--ServerApp.root_dir={ROOT}",
        f"--ServerApp.ip={args.host}",
        f"--ServerApp.port={args.port}",
        f"--IdentityProvider.token={args.token}",
    ]
    if not args.open_browser:
        command.append("--no-browser")

    run(command)


def wait_for_jupyter(proc: subprocess.Popen[str], args: argparse.Namespace) -> None:
    url = f"http://{args.host}:{args.port}/api?token={args.token}"
    print("Waiting for Jupyter API...")
    for _attempt in range(60):
        if proc.poll() is not None:
            print("Jupyter exited unexpectedly:")
            if LIVE_LOG.exists():
                print(LIVE_LOG.read_text(encoding="utf-8", errors="replace"))
            raise SystemExit(proc.returncode or 1)
        try:
            with urlopen(url, timeout=1) as response:
                if response.status < 500:
                    print("Jupyter is ready.")
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(1)

    print("Jupyter did not become ready:")
    if LIVE_LOG.exists():
        print(LIVE_LOG.read_text(encoding="utf-8", errors="replace"))
    raise SystemExit(1)


def live(args: argparse.Namespace) -> None:
    ensure_ready_for_notebooks()
    env = project_env()
    print(f"Starting Jupyter at http://{args.host}:{args.port}")

    with LIVE_LOG.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [str(part) for part in jupyter_server_command(args)],
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_jupyter(proc, args)
            run([venv_python(), "-m", "jupyter", "book", "start"], env=env)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def book(_args: argparse.Namespace) -> None:
    ensure_ready_for_notebooks()
    run([venv_python(), "-m", "jupyter", "book", "start"])


def jupyter(args: argparse.Namespace) -> None:
    ensure_ready_for_notebooks()
    run(jupyter_server_command(args))


def tikz(args: argparse.Namespace) -> None:
    tikz_dir = (ROOT / args.static_tikz_dir).resolve()
    tex_files = sorted(tikz_dir.glob("*.tex"))
    if not tex_files:
        print(f"No LaTeX figures found in {tikz_dir}")
        return

    for tex_file in tex_files:
        base = tex_file.with_suffix("")
        print(f"Compiling {tex_file.name}")
        subprocess.run(
            [
                args.latexmk,
                "-pdf",
                "-silent",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_file.name,
            ],
            cwd=tikz_dir,
            check=True,
        )
        subprocess.run(
            [args.pdftocairo, "-svg", f"{base.name}.pdf", f"{base.name}.svg"],
            cwd=tikz_dir,
            check=True,
        )
        subprocess.run(
            [args.latexmk, "-c", "-silent", tex_file.name],
            cwd=tikz_dir,
            stdout=subprocess.DEVNULL,
            check=True,
        )


def clean(_args: argparse.Namespace) -> None:
    for path in [LIVE_LOG]:
        if path.exists():
            path.unlink()

    for path in [
        ROOT / "_build",
        ROOT / ".jupyter_cache",
        ROOT / "__pycache__",
        CHAPTERS / "__pycache__",
        ROOT / "scripts" / "__pycache__",
        LOCAL_JUPYTER,
    ]:
        if path.exists():
            shutil.rmtree(path)

    print("Removed build output, caches, local Jupyter state, and live log.")


def clean_notebooks(_args: argparse.Namespace) -> None:
    notebooks_dir = configured_notebooks_dir(
        getattr(_args, "notebooks_dir", DEFAULT_NOTEBOOKS_DIR)
    )
    project_relative_path(notebooks_dir)
    notebooks_dir_resolved = notebooks_dir.resolve()
    root_resolved = ROOT.resolve()
    chapters_resolved = CHAPTERS.resolve()

    if notebooks_dir_resolved == root_resolved:
        raise SystemExit("Refusing to remove the project root as a notebooks directory.")

    removed_dir = False
    if notebooks_dir.exists() and notebooks_dir_resolved != chapters_resolved:
        shutil.rmtree(notebooks_dir)
        removed_dir = True

    removed_legacy = remove_legacy_chapter_notebooks()
    removed_checkpoints = remove_legacy_chapter_checkpoints()
    if removed_dir:
        print(f"Removed generated notebook folder {display_path(notebooks_dir)}.")
    print(f"Removed {removed_legacy} legacy generated notebooks from {display_path(CHAPTERS)}/.")
    if removed_checkpoints:
        print(f"Removed legacy checkpoints from {display_path(CHAPTERS / '.ipynb_checkpoints')}/.")


def clean_env(_args: argparse.Namespace) -> None:
    if venv_dir().exists():
        shutil.rmtree(venv_dir())
        print(f"Removed {display_path(venv_dir())}.")
    else:
        print(f"No environment found at {display_path(venv_dir())}.")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    help_parser = subparsers.add_parser("help")
    help_parser.set_defaults(func=public_help)

    install_parser = subparsers.add_parser("install")
    install_parser.set_defaults(func=install)

    notebooks_parser = subparsers.add_parser("notebooks")
    add_notebooks_dir_arg(notebooks_parser)
    notebooks_parser.set_defaults(func=notebooks)

    start_parser = subparsers.add_parser("start")
    add_server_args(start_parser)
    add_notebooks_dir_arg(start_parser)
    start_parser.add_argument("--open-browser", default="1", type=bool_arg)
    start_parser.set_defaults(func=start)

    live_parser = subparsers.add_parser("live")
    add_server_args(live_parser)
    live_parser.set_defaults(func=live)

    book_parser = subparsers.add_parser("book")
    book_parser.set_defaults(func=book)

    jupyter_parser = subparsers.add_parser("jupyter")
    add_server_args(jupyter_parser)
    jupyter_parser.set_defaults(func=jupyter)

    tikz_parser = subparsers.add_parser("tikz")
    tikz_parser.add_argument("--static-tikz-dir", default="chapters/_static/tikz")
    tikz_parser.add_argument("--latexmk", default="latexmk")
    tikz_parser.add_argument("--pdftocairo", default="pdftocairo")
    tikz_parser.set_defaults(func=tikz)

    clean_parser = subparsers.add_parser("clean")
    clean_parser.set_defaults(func=clean)

    clean_notebooks_parser = subparsers.add_parser("clean-notebooks")
    add_notebooks_dir_arg(clean_notebooks_parser)
    clean_notebooks_parser.set_defaults(func=clean_notebooks)

    clean_env_parser = subparsers.add_parser("clean-env")
    clean_env_parser.set_defaults(func=clean_env)

    return parser


def add_server_args(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument("--host", default="127.0.0.1")
    command_parser.add_argument("--port", default="8888")
    command_parser.add_argument("--token", default="myst-local")


def add_notebooks_dir_arg(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument("--notebooks-dir", default=DEFAULT_NOTEBOOKS_DIR)


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
