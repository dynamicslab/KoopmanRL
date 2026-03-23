import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_module(module, extra_args=None, timeout=300, env=None):
    cmd = ["uv", "run", "-m", module] + (extra_args or [])
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result
