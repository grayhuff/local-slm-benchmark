"""Best-effort hardware snapshot so results can be read in context.

A "tokens/sec" number means little without knowing it came from an M3 laptop
or an RTX 4090, so every saved run can carry this. All probes are optional
and fail quietly.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def _nvidia_gpus() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, total, used = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "vram_total_gb": round(float(total) / 1024, 1), "vram_used_gb": round(float(used) / 1024, 1)})
    return gpus


def system_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_name(),
        "python": platform.python_version(),
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        info.update(
            cpu_cores=psutil.cpu_count(logical=False),
            ram_total_gb=round(mem.total / 1e9, 1),
            ram_available_gb=round(mem.available / 1e9, 1),
            cpu_percent=psutil.cpu_percent(interval=None),
        )
    except ImportError:
        pass
    gpus = _nvidia_gpus()
    if gpus:
        info["gpus"] = gpus
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        info["gpus"] = [{"name": "Apple Silicon (unified memory)"}]
    return info
