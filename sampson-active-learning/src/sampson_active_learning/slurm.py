# src/sampson_active_learning/slurm.py

from pathlib import Path
from typing import Iterable

import subprocess


def render_basic_slurm_script(
    header_lines: Iterable[str],
    command_lines: Iterable[str],
) -> str:
    """
    Utility: concatenate header and body into a single batch script string.
    """
    header = list(header_lines)
    body = list(command_lines)
    return "\n".join(header + [""] + body) + "\n"


def write_and_submit(script_contents: str, script_path: Path) -> None:
    """
    Write script to disk and submit with sbatch.
    """
    script_path.write_text(script_contents)
    subprocess.run(["sbatch", str(script_path)], check=True)