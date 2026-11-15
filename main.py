from __future__ import annotations

import subprocess
import sys

from config import BASE_DIR


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(BASE_DIR / "app.py"),
        *sys.argv[1:],
    ]
    return subprocess.call(command, cwd=BASE_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
