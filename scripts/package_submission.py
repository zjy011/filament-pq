"""Build the plan's source-only ZIP (filament/, scripts/, configs/)."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/filament-pq-source.zip"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.out, "w", ZIP_DEFLATED) as bundle:
        for root in ("filament", "scripts", "configs"):
            for path in sorted(Path(root).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    bundle.write(path, path.as_posix())
    with ZipFile(args.out) as bundle:
        assert bundle.testzip() is None
        assert all(
            name.split("/")[0] in {"filament", "scripts", "configs"} for name in bundle.namelist()
        )
        assert "filament/unet/LICENSE.clDice.txt" in bundle.namelist()
        count = len(bundle.namelist())
    print(f"{args.out}: {count} files; SHA256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
