"""Convert one or more .m4a files to PCM .wav using ffmpeg.

Examples:
    python convert_m4a_to_wav.py recording.m4a
    python convert_m4a_to_wav.py recordings/ --output recordings/wav
    python convert_m4a_to_wav.py take.m4a --sr 22050 --mono
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="m4a files or directories containing m4a files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output wav path for a single input, or a directory for multiple inputs",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="resample to this rate in Hz (default: keep the source rate)",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="downmix to one channel",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing wav file",
    )
    return parser.parse_args()


def collect_m4a_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.glob("*.m4a")))
            files.extend(sorted(path.glob("*.M4A")))
        elif path.is_file():
            if path.suffix.lower() != ".m4a":
                raise SystemExit(f"Not an m4a file: {path}")
            files.append(path)
        else:
            raise SystemExit(f"Path not found: {path}")

    unique = list(dict.fromkeys(files))
    if not unique:
        raise SystemExit("No m4a files found.")
    return unique


def output_path_for(source: Path, output: Path | None, multiple: bool) -> Path:
    if output is None:
        return source.with_suffix(".wav")
    if multiple or output.is_dir() or output.suffix.lower() != ".wav":
        return output / f"{source.stem}.wav"
    return output


def convert_one(
    source: Path,
    destination: Path,
    sample_rate: int | None,
    mono: bool,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite): {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    command.extend(["-i", str(source)])
    if mono:
        command.extend(["-ac", "1"])
    if sample_rate is not None:
        command.extend(["-ar", str(sample_rate)])
    command.extend(["-c:a", "pcm_s16le", str(destination)])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(f"ffmpeg failed for {source}:\n{detail}")

    print(f"{source} -> {destination}")


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found on PATH.")

    args = parse_args()
    sources = collect_m4a_files(args.inputs)
    multiple = len(sources) > 1

    if args.output is not None and not multiple and args.output.suffix.lower() == ".wav":
        destinations = [args.output]
    else:
        destinations = [
            output_path_for(source, args.output, multiple) for source in sources
        ]

    for source, destination in zip(sources, destinations):
        convert_one(source, destination, args.sr, args.mono, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
