#!/usr/bin/env python3
"""Verify and restore the complete, lossless GitHub data snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    backup = root / "data" / "archive"
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    directories = manifest["directories"]
    version = manifest.get("format_version", 1)
    if version == 1 and directories == ["nero_demos", "nero_aligned_v1"]:
        destinations = {"nero_demos": root / "data/nero_demos/yellow",
                        "nero_aligned_v1": root / "data/nero_aligned_v1"}
    elif version == 2 and directories == ["nero_demos", "nero_aligned_v1", "nero_aligned_red_v1"]:
        destinations = {name: root / "data" / name for name in directories}
    else:
        raise SystemExit("Unexpected data directories or format version in manifest")
    if not args.verify_only:
        for target in destinations.values():
            if target.exists() or target.is_symlink():
                raise SystemExit(f"Refusing to overwrite existing directory: {target}")

    with tempfile.TemporaryDirectory(prefix="nero-data-", dir=root / "data") as temp:
        work = Path(temp)
        archive = work / "nero-data.tar.gz"
        full_hash = hashlib.sha256()
        total = 0
        with archive.open("wb") as output:
            for part in manifest["parts"]:
                name = part["name"]
                if Path(name).name != name:
                    raise SystemExit(f"Invalid part name: {name}")
                digest = hashlib.sha256()
                size = 0
                with (backup / name).open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        full_hash.update(block)
                        size += len(block)
                if size != part["size"] or digest.hexdigest() != part["sha256"]:
                    raise SystemExit(f"Checksum mismatch: {name}")
                total += size
        if total != manifest["archive_size"] or full_hash.hexdigest() != manifest["archive_sha256"]:
            raise SystemExit("Archive checksum mismatch")
        print(f"Verified {len(manifest['parts'])} parts ({total:,} bytes).")
        if args.verify_only:
            return

        extracted = work / "extracted"
        count = 0
        original_bytes = 0
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or len(path.parts) < 2
                    or path.parts[0] != "data"
                    or path.parts[1] not in directories
                ):
                    raise SystemExit(f"Unexpected archive path: {member.name}")
                target = extracted.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = source.extractfile(member)
                    if content is None:
                        raise SystemExit(f"Missing archive content: {member.name}")
                    with content, target.open("xb") as output:
                        shutil.copyfileobj(content, output)
                    count += 1
                    original_bytes += member.size
                else:
                    raise SystemExit(f"Unsupported archive entry: {member.name}")
        if count != manifest["original_file_count"] or original_bytes != manifest["original_bytes"]:
            raise SystemExit("Restored file count or byte count mismatch")
        for name in directories:
            target = destinations[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            (extracted / "data" / name).rename(target)
        print(f"Restored {count:,} files ({original_bytes:,} bytes) to {root / 'data'}")


if __name__ == "__main__":
    main()
