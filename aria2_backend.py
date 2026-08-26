"""Optional aria2c backend for downloading one HTTP(S) object.

The registry client remains responsible for authentication and URL resolution.
This module deliberately knows nothing about Docker/Ollama registry semantics;
it only manages the aria2c subprocess and its resumable output file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Optional


class Aria2Error(RuntimeError):
    """Base error for the optional aria2 backend."""


class Aria2UnavailableError(Aria2Error):
    """aria2c was requested but is not available."""


class Aria2DownloadError(Aria2Error):
    """aria2c failed to download the requested object."""


class Aria2Backend:
    """Small wrapper around the optional aria2c executable."""

    def __init__(self, executable: Optional[str] = None) -> None:
        self.executable = executable or shutil.which("aria2c")

    @property
    def available(self) -> bool:
        return bool(self.executable)

    @staticmethod
    def control_path(output_path: Path) -> Path:
        """Return aria2's control-file path for an output file."""
        return Path(f"{output_path}.aria2")

    @classmethod
    def remove_control_file(cls, output_path: Path) -> None:
        """Remove stale aria2 state without affecting the downloaded bytes."""
        cls.control_path(output_path).unlink(missing_ok=True)

    def _require_available(self) -> str:
        if not self.executable:
            raise Aria2UnavailableError(
                "aria2c was not found on PATH; install aria2 or use "
                "--backend httpx"
            )

        return self.executable

    @staticmethod
    def _max_tries(retries: int) -> int:
        # aria2 uses 0 for unlimited attempts. The downloader's retry count
        # describes retries after the first attempt, so add one here.
        if retries < 0:
            return 0

        return max(1, retries + 1)

    @staticmethod
    def _validate_headers(headers: Mapping[str, str]) -> Dict[str, str]:
        validated: Dict[str, str] = {}

        for name, value in headers.items():
            if not name or "\r" in name or "\n" in name:
                raise ValueError("invalid HTTP header name for aria2")

            if "\r" in value or "\n" in value:
                raise ValueError(
                    f"invalid HTTP header value for aria2: {name}"
                )

            validated[str(name)] = str(value)

        return validated

    def download(
        self,
        url: str,
        output_path: Path,
        *,
        expected_size: int,
        expected_hash: Optional[str],
        headers: Optional[Mapping[str, str]],
        connections: int,
        min_split_size: int,
        retries: int,
        connect_timeout: float,
        read_timeout: float,
    ) -> None:
        """Download one object into ``output_path`` using aria2c.

        The output and its ``.aria2`` control file are intentionally left in
        place when aria2c fails so a later invocation can resume the transfer.
        The caller owns final digest verification and installation.
        """
        executable = self._require_available()

        if connections < 1:
            raise ValueError("aria2 connections must be at least 1")

        if min_split_size < 1:
            raise ValueError("aria2 minimum split size must be positive")

        if expected_size < 1:
            raise ValueError("aria2 expected size must be positive")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            executable,
            "--no-conf",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--continue=true",
            "--file-allocation=none",
            "--enable-color=false",
            "--show-console-readout=true",
            "--summary-interval=1",
            "--console-log-level=warn",
            "--download-result=hide",
            f"--split={connections}",
            f"--max-connection-per-server={connections}",
            f"--min-split-size={min_split_size}",
            f"--max-tries={self._max_tries(retries)}",
            "--retry-wait=2",
            f"--connect-timeout={connect_timeout:g}",
            f"--timeout={read_timeout:g}",
            "--check-certificate=true",
            "--check-integrity=true",
            "--allow-piece-length-change=true",
            f"--dir={output_path.parent}",
            f"--out={output_path.name}",
        ]

        if expected_hash:
            command.append(f"--checksum=sha-256={expected_hash}")

        for name, value in self._validate_headers(headers or {}).items():
            command.append(f"--header={name}: {value}")

        command.append(url)

        try:
            completed = subprocess.run(command, check=False)
        except OSError as exc:
            raise Aria2DownloadError(
                f"could not start aria2c: {exc}"
            ) from exc

        if completed.returncode != 0:
            raise Aria2DownloadError(
                f"aria2c exited with status {completed.returncode}"
            )

        if not output_path.exists():
            raise Aria2DownloadError(
                f"aria2c completed without creating {output_path.name}"
            )

        actual_size = output_path.stat().st_size

        if actual_size != expected_size:
            raise Aria2DownloadError(
                f"aria2c produced {actual_size} bytes; "
                f"expected {expected_size}"
            )

