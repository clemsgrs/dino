import argparse
import subprocess
from pathlib import Path
from natsort import natsorted


def infer_entries(source_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    tarball_files = natsorted(list(Path(source_dir).glob("*.tar")))
    for fp in tarball_files:
        stem = fp.stem
        cmd = [
            "/usr/local/bin/python3",
            "infer_entries_new.py",
            "--tarball_path",
            f"{fp}",
            "--output_root",
            f"{output_dir}",
            "--name",
            f"{stem}",
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create multiple tarballs from files in a folder, optimized for large file sets."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source tarball file directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output directory where the npy files will be saved.",
    )
    args = parser.parse_args()

    infer_entries(args.source, args.output)
