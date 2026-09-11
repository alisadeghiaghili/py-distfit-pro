"""Contracts for deterministic archive canonicalisation."""

from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.build_reproducible import _normalise_sdist, _normalise_wheel

EPOCH = 1_789_084_800


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproducibleArchiveTests(unittest.TestCase):
    def test_sdist_metadata_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs: list[Path] = []
            for index, mtime in enumerate((1_700_000_000, 1_800_000_000)):
                raw = root / f"raw-{index}.tar.gz"
                with tarfile.open(raw, "w:gz") as archive:
                    info = tarfile.TarInfo("package/value.txt")
                    info.size = 5
                    info.mtime = mtime
                    info.uid = index + 1
                    archive.addfile(info, io.BytesIO(b"value"))
                target = root / f"normal-{index}.tar.gz"
                _normalise_sdist(raw, target, EPOCH)
                outputs.append(target)
            self.assertEqual(_digest(outputs[0]), _digest(outputs[1]))

    def test_wheel_metadata_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs: list[Path] = []
            for index, stamp in enumerate(((2020, 1, 1, 0, 0, 0), (2025, 1, 1, 0, 0, 0))):
                raw = root / f"raw-{index}.whl"
                with zipfile.ZipFile(raw, "w") as archive:
                    info = zipfile.ZipInfo("package/value.txt", date_time=stamp)
                    archive.writestr(info, b"value")
                    archive.writestr(
                        zipfile.ZipInfo("package-1.0.dist-info/WHEEL", date_time=stamp),
                        f"Wheel-Version: 1.0\nGenerator: backend-{index}\n",
                    )
                    archive.writestr(
                        zipfile.ZipInfo("package-1.0.dist-info/RECORD", date_time=stamp), b""
                    )
                target = root / f"normal-{index}.whl"
                _normalise_wheel(raw, target, EPOCH)
                outputs.append(target)
            self.assertEqual(_digest(outputs[0]), _digest(outputs[1]))


if __name__ == "__main__":
    unittest.main()
