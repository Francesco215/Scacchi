from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    subprocess.run(["bash", str(root_dir / "scripts" / "install_mohex.sh")], check=True)


if __name__ == "__main__":
    main()
