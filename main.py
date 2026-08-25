#!/usr/bin/env python3
"""
Ollama Model Downloader v4

Cross-platform, resumable, multi-connection downloader for Ollama-compatible
registries.

v4 adds aria2-style parallel downloading:
  - A large blob is divided into independent contiguous segments.
  - Multiple worker threads download those segments simultaneously.
  - Every worker uses bounded HTTP Range requests.
  - Every segment has its own resumable .part file.
  - Existing v3 sequential .part files are migrated into v4 segments.
  - Segments are merged only after all workers finish.
  - The merged blob is SHA-256 verified before Ollama sees it.

Dependencies:
  httpx
  tenacity
  filelock
  rich
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import httpx
from filelock import FileLock, Timeout as FileLockTimeout
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from tenacity import (
    Retrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    stop_never,
    wait_random_exponential,
)


DEFAULT_REGISTRY = "https://registry.ollama.ai"

STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB disk/network streaming buffer
DEFAULT_RANGE_SIZE_MIB = 8
DEFAULT_CONNECTIONS = 8
DEFAULT_RETRIES = 10
DEFAULT_CONNECT_TIMEOUT = 20.0
DEFAULT_READ_TIMEOUT = 20.0
DEFAULT_POOL_TIMEOUT = 20.0

USER_AGENT = "ollama-cross-platform-downloader/4.0"

RETRYABLE_HTTP_STATUSES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}

_AUTH_RE = re.compile(r'(\w+)="([^"]*)"')
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.I)

console = Console()


class RetryableHTTPStatus(httpx.HTTPStatusError):
    """Temporary HTTP status that should be retried."""


class RetryableRangeError(OSError):
    """Malformed, truncated, or otherwise retryable Range response."""


class RangeNotSupportedError(RuntimeError):
    """Server ignored Range semantics, making parallel resume unsafe."""


@dataclass(frozen=True)
class Segment:
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{n} B"


def default_models_dir() -> Path:
    configured = os.environ.get("OLLAMA_MODELS")

    if configured:
        return Path(configured).expanduser()

    return Path.home() / ".ollama" / "models"


def parse_model_ref(ref: str) -> Tuple[str, str]:
    ref = ref.strip().strip("/")

    if not ref:
        raise ValueError("empty model reference")

    for prefix in (
        "https://registry.ollama.ai/",
        "http://registry.ollama.ai/",
        "registry.ollama.ai/",
    ):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            break

    last_component = ref.rsplit("/", 1)[-1]

    if ":" in last_component:
        name, tag = ref.rsplit(":", 1)
    else:
        name, tag = ref, "latest"

    if not name or not tag:
        raise ValueError(f"invalid model reference: {ref!r}")

    repo = name if "/" in name else f"library/{name}"
    return repo, tag


def digest_filename(digest: str) -> str:
    algo, sep, value = digest.partition(":")

    if not sep or not algo or not value:
        raise ValueError(f"invalid digest: {digest}")

    return f"{algo}-{value}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def build_segments(
    total_size: int,
    requested_connections: int,
    range_size: int,
) -> List[Segment]:
    """
    Split a blob into contiguous download lanes.

    Tiny blobs do not get pointless extra connections. A blob must contain
    at least roughly one range window per active worker.
    """
    if total_size <= 0:
        raise ValueError("total_size must be positive")

    useful_workers = max(1, math.ceil(total_size / range_size))
    worker_count = min(requested_connections, useful_workers)

    base = total_size // worker_count
    remainder = total_size % worker_count

    segments: List[Segment] = []
    cursor = 0

    for index in range(worker_count):
        size = base + (1 if index < remainder else 0)
        start = cursor
        end = start + size - 1
        segments.append(Segment(index=index, start=start, end=end))
        cursor = end + 1

    if cursor != total_size:
        raise RuntimeError("internal segment layout error")

    return segments


def segment_path(final_path: Path, segment: Segment) -> Path:
    return Path(
        f"{final_path}.segment-{segment.index:03d}.part"
    )


class RegistryClient:
    def __init__(
        self,
        base_url: str,
        repo: str,
        *,
        connect_timeout: float,
        read_timeout: float,
        pool_timeout: float,
        max_connections: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.repo = repo
        self.token: Optional[str] = None
        self._auth_lock = threading.Lock()

        self.client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=30.0,
                pool=pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=max(max_connections, 4),
                max_keepalive_connections=max(max_connections, 4),
                keepalive_expiry=20.0,
            ),
            headers={
                "User-Agent": USER_AGENT,
                # Exact bytes are required for digest verification.
                "Accept-Encoding": "identity",
            },
        )

    def __enter__(self) -> "RegistryClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.client.close()

    def _authorization_headers(self) -> Dict[str, str]:
        token = self.token

        if token:
            return {"Authorization": f"Bearer {token}"}

        return {}

    def _get_bearer_token(self, challenge: str) -> str:
        params = dict(_AUTH_RE.findall(challenge))
        realm = params.get("realm")

        if not realm:
            raise RuntimeError(
                "registry requested Bearer authentication but supplied no realm"
            )

        query: Dict[str, str] = {}

        if params.get("service"):
            query["service"] = params["service"]

        query["scope"] = params.get("scope") or f"repository:{self.repo}:pull"

        separator = "&" if "?" in realm else "?"
        response = self.client.get(realm + separator + urlencode(query))
        response.raise_for_status()

        payload = response.json()
        token = payload.get("token") or payload.get("access_token")

        if not token:
            raise RuntimeError("registry token response contained no token")

        return token

    def _refresh_token(self, challenge: str) -> None:
        with self._auth_lock:
            self.token = self._get_bearer_token(challenge)

    def _send(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        merged = self._authorization_headers()

        if headers:
            merged.update(headers)

        if stream:
            request = self.client.build_request(
                method,
                url,
                headers=merged,
            )
            return self.client.send(request, stream=True)

        return self.client.request(
            method,
            url,
            headers=merged,
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        response = self._send(
            method,
            url,
            headers=headers,
            stream=stream,
        )

        if response.status_code == 401:
            challenge = response.headers.get("WWW-Authenticate", "")
            response.close()

            if challenge.lower().startswith("bearer "):
                self._refresh_token(challenge)
                response = self._send(
                    method,
                    url,
                    headers=headers,
                    stream=stream,
                )

        if response.status_code in RETRYABLE_HTTP_STATUSES:
            status = response.status_code
            reason = response.reason_phrase
            request = response.request
            response.close()

            raise RetryableHTTPStatus(
                f"temporary HTTP {status} {reason}",
                request=request,
                response=httpx.Response(
                    status,
                    request=request,
                    reason_phrase=reason,
                ),
            )

        response.raise_for_status()
        return response

    def get_manifest(self, tag: str) -> dict:
        url = f"{self.base_url}/v2/{self.repo}/manifests/{tag}"

        response = self.request(
            "GET",
            url,
            headers={
                "Accept": (
                    "application/vnd.docker.distribution.manifest.v2+json,"
                    "application/vnd.oci.image.manifest.v1+json,"
                    "application/json"
                )
            },
        )

        try:
            return response.json()
        finally:
            response.close()

    def open_blob_range(
        self,
        digest: str,
        start: int,
        end: int,
    ) -> httpx.Response:
        return self.request(
            "GET",
            f"{self.base_url}/v2/{self.repo}/blobs/{digest}",
            headers={
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
            },
            stream=True,
        )


RETRYABLE_EXCEPTIONS = (
    RetryableHTTPStatus,
    RetryableRangeError,
    httpx.TransportError,
    OSError,
)


def retry_message(retry_state: RetryCallState) -> None:
    delay = retry_state.next_action.sleep if retry_state.next_action else 0.0
    exc = retry_state.outcome.exception() if retry_state.outcome else None

    console.print(
        f"  [yellow]Retrying in {delay:.1f}s[/]"
        + (f" — {exc}" if exc else "")
    )


def make_retryer(retries: int) -> Retrying:
    if retries < 0:
        stop_rule = stop_never
    else:
        stop_rule = stop_after_attempt(retries + 1)

    return Retrying(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_random_exponential(
            multiplier=1,
            max=30,
        ),
        stop=stop_rule,
        before_sleep=retry_message,
        reraise=True,
    )


def fetch_manifest_with_retry(
    client: RegistryClient,
    tag: str,
    retries: int,
) -> dict:
    result: Optional[dict] = None

    for attempt in make_retryer(retries):
        with attempt:
            result = client.get_manifest(tag)

    if result is None:
        raise RuntimeError("manifest download returned no result")

    return result


def manifest_objects(manifest: dict) -> Iterable[dict]:
    config = manifest.get("config")

    if isinstance(config, dict):
        yield config

    layers = manifest.get("layers")

    if isinstance(layers, list):
        yield from layers


def prepare_existing_final_blob(
    final_path: Path,
    legacy_part_path: Path,
    expected_size: int,
    expected_hash: str,
) -> bool:
    """
    Return True when a valid complete final blob already exists.

    Incomplete old-style final files are converted to the v3/v4 legacy .part
    path so they can be migrated into v4 segments.
    """
    if not final_path.exists():
        return False

    local_size = final_path.stat().st_size

    if local_size == expected_size:
        console.print("  Existing blob found; verifying SHA-256...")

        if sha256_file(final_path) == expected_hash:
            console.print("  [green]Already complete and valid — skipped.[/]")
            return True

        console.print("  [yellow]Checksum mismatch — redownloading.[/]")
        final_path.unlink(missing_ok=True)
        return False

    if local_size < expected_size and not legacy_part_path.exists():
        console.print(
            f"  Migrating incomplete final blob "
            f"({human_bytes(local_size)}) to resumable state"
        )
        final_path.replace(legacy_part_path)
    else:
        final_path.unlink(missing_ok=True)

    return False


def migrate_legacy_part(
    legacy_part_path: Path,
    final_path: Path,
    segments: List[Segment],
    expected_size: int,
) -> None:
    """
    Convert a v3 sequential `.part` file (bytes 0..N) into v4 per-segment
    partial files. This lets users upgrade without losing already-downloaded
    data.
    """
    if not legacy_part_path.exists():
        return

    legacy_size = legacy_part_path.stat().st_size

    if legacy_size <= 0:
        legacy_part_path.unlink(missing_ok=True)
        return

    if legacy_size > expected_size:
        console.print(
            "  [yellow]Legacy .part is larger than the blob; discarding it.[/]"
        )
        legacy_part_path.unlink(missing_ok=True)
        return

    # Do not overwrite newer v4 segment state.
    existing_segment_state = any(
        segment_path(final_path, segment).exists()
        for segment in segments
    )

    if existing_segment_state:
        console.print(
            "  [yellow]Both legacy and v4 partial state exist; "
            "keeping v4 segment state and leaving legacy .part untouched.[/]"
        )
        return

    console.print(
        f"  [cyan]Migrating legacy resume data: "
        f"{human_bytes(legacy_size)}[/]"
    )

    with legacy_part_path.open("rb") as source:
        for segment in segments:
            if segment.start >= legacy_size:
                break

            bytes_available = min(
                segment.size,
                legacy_size - segment.start,
            )

            if bytes_available <= 0:
                continue

            source.seek(segment.start)

            destination = segment_path(final_path, segment)

            with destination.open("wb") as out:
                remaining = bytes_available

                while remaining:
                    chunk = source.read(
                        min(4 * 1024 * 1024, remaining)
                    )

                    if not chunk:
                        raise RuntimeError(
                            "unexpected EOF while migrating legacy .part"
                        )

                    out.write(chunk)
                    remaining -= len(chunk)

    legacy_part_path.unlink(missing_ok=True)


def sanitize_segment_files(
    final_path: Path,
    segments: List[Segment],
) -> int:
    """
    Remove impossible segment files and return total valid resumed bytes.
    """
    resumed = 0

    for segment in segments:
        path = segment_path(final_path, segment)

        if not path.exists():
            continue

        size = path.stat().st_size

        if size > segment.size:
            console.print(
                f"  [yellow]Segment {segment.index} is too large; "
                "restarting that segment.[/]"
            )
            path.unlink(missing_ok=True)
            continue

        resumed += size

    return resumed


def parse_content_range(
    value: str,
) -> Tuple[int, int, Optional[int]]:
    match = _CONTENT_RANGE_RE.match(value.strip())

    if not match:
        raise RetryableRangeError(
            f"invalid or missing Content-Range header: {value!r}"
        )

    start = int(match.group(1))
    end = int(match.group(2))
    total_text = match.group(3)
    total = None if total_text == "*" else int(total_text)

    return start, end, total


def download_segment_range_once(
    client: RegistryClient,
    digest: str,
    expected_blob_size: int,
    segment: Segment,
    path: Path,
    range_size: int,
    progress: Progress,
    progress_task: int,
    stop_event: threading.Event,
) -> int:
    """
    Download one bounded sub-range of one segment.

    The local segment size is re-read on every retry, so if the connection
    drops after writing some bytes, the next request begins at the exact byte
    already persisted in that segment file.
    """
    if stop_event.is_set():
        return 0

    local_size = path.stat().st_size if path.exists() else 0

    if local_size >= segment.size:
        return 0

    global_start = segment.start + local_size
    requested_end = min(
        segment.end,
        global_start + range_size - 1,
    )

    response = client.open_blob_range(
        digest,
        global_start,
        requested_end,
    )

    try:
        if response.status_code != 206:
            raise RangeNotSupportedError(
                f"registry/CDN returned HTTP {response.status_code} "
                "instead of 206 for a ranged request"
            )

        content_range = response.headers.get("Content-Range", "")
        response_start, response_end, response_total = parse_content_range(
            content_range
        )

        if response_start != global_start:
            raise RetryableRangeError(
                f"wrong Content-Range start: "
                f"{response_start} != {global_start}"
            )

        if response_end > requested_end:
            raise RetryableRangeError(
                f"server returned beyond requested end: "
                f"{response_end} > {requested_end}"
            )

        if response_end < response_start:
            raise RetryableRangeError(
                "server returned an invalid Content-Range"
            )

        if (
            response_total is not None
            and response_total != expected_blob_size
        ):
            raise RetryableRangeError(
                f"wrong total blob size in Content-Range: "
                f"{response_total} != {expected_blob_size}"
            )

        expected_response_bytes = response_end - response_start + 1
        bytes_written = 0

        with path.open("ab") as out:
            for chunk in response.iter_raw(STREAM_CHUNK_SIZE):
                if stop_event.is_set():
                    raise RuntimeError("download cancelled")

                if not chunk:
                    continue

                remaining = expected_response_bytes - bytes_written

                if len(chunk) > remaining:
                    raise RetryableRangeError(
                        "server sent more data than Content-Range declared"
                    )

                out.write(chunk)
                bytes_written += len(chunk)
                progress.advance(progress_task, len(chunk))

            out.flush()

        if bytes_written != expected_response_bytes:
            raise RetryableRangeError(
                f"truncated range: got {bytes_written} bytes, "
                f"expected {expected_response_bytes}"
            )

        expected_local_size = local_size + expected_response_bytes
        actual_local_size = path.stat().st_size

        if actual_local_size != expected_local_size:
            raise RetryableRangeError(
                f"segment file size mismatch: "
                f"{actual_local_size} != {expected_local_size}"
            )

        return bytes_written

    finally:
        response.close()


def download_segment_worker(
    client: RegistryClient,
    digest: str,
    expected_blob_size: int,
    final_path: Path,
    segment: Segment,
    range_size: int,
    retries: int,
    progress: Progress,
    progress_task: int,
    stop_event: threading.Event,
) -> None:
    path = segment_path(final_path, segment)

    while not stop_event.is_set():
        local_size = path.stat().st_size if path.exists() else 0

        if local_size >= segment.size:
            return

        downloaded: Optional[int] = None

        for attempt in make_retryer(retries):
            with attempt:
                downloaded = download_segment_range_once(
                    client,
                    digest,
                    expected_blob_size,
                    segment,
                    path,
                    range_size,
                    progress,
                    progress_task,
                    stop_event,
                )

        if downloaded is None:
            raise RuntimeError(
                f"segment {segment.index} downloader returned no result"
            )

        if downloaded <= 0:
            # It may have become complete between checks.
            if path.exists() and path.stat().st_size == segment.size:
                return

            raise RuntimeError(
                f"segment {segment.index} made no forward progress"
            )


def merge_and_verify_segments(
    final_path: Path,
    legacy_part_path: Path,
    segments: List[Segment],
    expected_size: int,
    expected_hash: str,
) -> None:
    """
    Merge completed segment files in order and calculate SHA-256 while merging.
    Segment files are not deleted until verification succeeds.
    """
    merge_path = Path(f"{final_path}.merge.part")
    merge_path.unlink(missing_ok=True)

    digest = hashlib.sha256()
    total_written = 0

    console.print("  Merging parallel segments...")

    with merge_path.open("wb") as out:
        for segment in segments:
            path = segment_path(final_path, segment)

            if not path.exists():
                raise RuntimeError(
                    f"segment {segment.index} is missing before merge"
                )

            size = path.stat().st_size

            if size != segment.size:
                raise RuntimeError(
                    f"segment {segment.index} incomplete before merge: "
                    f"{size} != {segment.size}"
                )

            with path.open("rb") as source:
                while True:
                    chunk = source.read(4 * 1024 * 1024)

                    if not chunk:
                        break

                    out.write(chunk)
                    digest.update(chunk)
                    total_written += len(chunk)

    if total_written != expected_size:
        merge_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"merged blob size mismatch: "
            f"{total_written} != {expected_size}"
        )

    console.print("  Verifying SHA-256...")
    actual_hash = digest.hexdigest()

    if actual_hash != expected_hash:
        corrupt_path = Path(f"{final_path}.corrupt")
        corrupt_path.unlink(missing_ok=True)
        os.replace(merge_path, corrupt_path)

        # Segment data cannot be trusted if the complete digest failed.
        for segment in segments:
            segment_path(final_path, segment).unlink(missing_ok=True)

        legacy_part_path.unlink(missing_ok=True)

        raise RuntimeError(
            f"SHA-256 mismatch: got {actual_hash}, "
            f"expected {expected_hash}. "
            f"Corrupt merged blob saved as {corrupt_path.name}; "
            "segments were cleared so the next run redownloads them."
        )

    os.replace(merge_path, final_path)

    for segment in segments:
        segment_path(final_path, segment).unlink(missing_ok=True)

    legacy_part_path.unlink(missing_ok=True)

    console.print("  [bold green]OK[/]")


def download_blob(
    client: RegistryClient,
    digest: str,
    expected_size: int,
    blobs_dir: Path,
    retries: int,
    lock_timeout: float,
    range_size: int,
    connections: int,
) -> None:
    algorithm, _, expected_hash = digest.partition(":")

    if algorithm != "sha256":
        raise RuntimeError(
            f"unsupported digest algorithm: {algorithm}"
        )

    final_path = blobs_dir / digest_filename(digest)
    legacy_part_path = Path(f"{final_path}.part")
    lock_path = Path(f"{final_path}.lock")

    try:
        with FileLock(str(lock_path), timeout=lock_timeout):
            if prepare_existing_final_blob(
                final_path,
                legacy_part_path,
                expected_size,
                expected_hash,
            ):
                return

            segments = build_segments(
                expected_size,
                connections,
                range_size,
            )

            migrate_legacy_part(
                legacy_part_path,
                final_path,
                segments,
                expected_size,
            )

            resumed = sanitize_segment_files(
                final_path,
                segments,
            )

            active_workers = len(segments)

            console.print(
                f"  [cyan]Connections for this blob:[/] {active_workers}"
            )

            if resumed:
                console.print(
                    f"  [cyan]Resuming:[/] "
                    f"{human_bytes(resumed)} / {human_bytes(expected_size)}"
                )

            progress = Progress(
                TextColumn("  [progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            )

            stop_event = threading.Event()

            with progress:
                task = progress.add_task(
                    digest[:20] + "…",
                    total=expected_size,
                    completed=resumed,
                )

                executor = ThreadPoolExecutor(
                    max_workers=active_workers,
                    thread_name_prefix="ollama-range",
                )

                futures: List[Future[None]] = []

                try:
                    for segment in segments:
                        path = segment_path(final_path, segment)

                        if (
                            path.exists()
                            and path.stat().st_size == segment.size
                        ):
                            continue

                        futures.append(
                            executor.submit(
                                download_segment_worker,
                                client,
                                digest,
                                expected_size,
                                final_path,
                                segment,
                                range_size,
                                retries,
                                progress,
                                task,
                                stop_event,
                            )
                        )

                    for future in as_completed(futures):
                        future.result()

                except Exception:
                    stop_event.set()

                    for future in futures:
                        future.cancel()

                    raise

                finally:
                    executor.shutdown(
                        wait=True,
                        cancel_futures=True,
                    )

            merge_and_verify_segments(
                final_path,
                legacy_part_path,
                segments,
                expected_size,
                expected_hash,
            )

    except FileLockTimeout as exc:
        raise RuntimeError(
            f"could not obtain lock for {final_path.name}; "
            "another downloader may already be downloading this blob"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reliable cross-platform multi-connection Ollama model downloader."
        )
    )

    parser.add_argument(
        "model",
        help="Model reference, e.g. qwen3.8:27b",
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_models_dir(),
        help=(
            "Ollama models directory "
            "($OLLAMA_MODELS or ~/.ollama/models)"
        ),
    )

    parser.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help=f"Registry URL (default: {DEFAULT_REGISTRY})",
    )

    parser.add_argument(
        "--connections",
        type=int,
        default=DEFAULT_CONNECTIONS,
        help=(
            "Maximum simultaneous HTTP Range connections per blob "
            f"(default: {DEFAULT_CONNECTIONS})"
        ),
    )

    parser.add_argument(
        "--range-size-mib",
        type=int,
        default=DEFAULT_RANGE_SIZE_MIB,
        help=(
            "Maximum size of each individual Range request in MiB "
            f"(default: {DEFAULT_RANGE_SIZE_MIB})"
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=(
            "Retries per failed Range request "
            f"(default: {DEFAULT_RETRIES}; -1 = unlimited)"
        ),
    )

    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        help=(
            "Connection timeout in seconds "
            f"(default: {DEFAULT_CONNECT_TIMEOUT:g})"
        ),
    )

    parser.add_argument(
        "--read-timeout",
        type=float,
        default=DEFAULT_READ_TIMEOUT,
        help=(
            "Read inactivity timeout in seconds "
            f"(default: {DEFAULT_READ_TIMEOUT:g})"
        ),
    )

    parser.add_argument(
        "--pool-timeout",
        type=float,
        default=DEFAULT_POOL_TIMEOUT,
        help=(
            "HTTP connection-pool timeout in seconds "
            f"(default: {DEFAULT_POOL_TIMEOUT:g})"
        ),
    )

    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a blob lock (default: 5)",
    )

    args = parser.parse_args()

    if not 1 <= args.connections <= 32:
        parser.error("--connections must be between 1 and 32")

    if not 1 <= args.range_size_mib <= 1024:
        parser.error("--range-size-mib must be between 1 and 1024")

    if args.retries < -1:
        parser.error("--retries must be -1 or greater")

    for name in (
        "connect_timeout",
        "read_timeout",
        "pool_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be > 0"
            )

    if args.lock_timeout < 0:
        parser.error("--lock-timeout must be >= 0")

    try:
        repo, tag = parse_model_ref(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    registry = args.registry.rstrip("/")
    parsed_registry = urlparse(registry)

    if not parsed_registry.scheme or not parsed_registry.netloc:
        parser.error("--registry must be a full URL")

    models_dir = args.models_dir.expanduser().resolve()
    blobs_dir = models_dir / "blobs"

    manifest_path = (
        models_dir
        / "manifests"
        / parsed_registry.netloc
        / Path(*repo.split("/"))
        / tag
    )

    blobs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    range_size = args.range_size_mib * 1024 * 1024

    console.print(f"[bold]Model:[/]        {repo}:{tag}")
    console.print(f"[bold]Registry:[/]     {registry}")
    console.print(f"[bold]Models dir:[/]   {models_dir}")
    console.print(f"[bold]Connections:[/]  {args.connections}")
    console.print(f"[bold]Range size:[/]    {args.range_size_mib} MiB")
    console.print(
        f"[bold]Read timeout:[/]  {args.read_timeout:g}s inactivity"
    )
    console.print(
        f"[bold]Retries:[/]       "
        + ("unlimited" if args.retries < 0 else str(args.retries))
    )
    console.print()

    with RegistryClient(
        registry,
        repo,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        pool_timeout=args.pool_timeout,
        # Extra headroom for auth/manifest requests while all workers run.
        max_connections=args.connections + 4,
    ) as client:
        console.print("[bold]Fetching manifest...[/]")

        manifest = fetch_manifest_with_retry(
            client,
            tag,
            args.retries,
        )

        blobs: Dict[str, int] = {}

        for obj in manifest_objects(manifest):
            digest = obj.get("digest")

            if not digest:
                continue

            size = int(obj.get("size") or 0)

            if size <= 0:
                raise RuntimeError(
                    f"invalid size for {digest}: {size}"
                )

            blobs.setdefault(digest, size)

        if not blobs:
            raise RuntimeError(
                "manifest contains no downloadable blobs"
            )

        console.print(f"[bold]Blobs:[/]        {len(blobs)}")
        console.print(
            f"[bold]Total size:[/]   "
            f"{human_bytes(sum(blobs.values()))}"
        )
        console.print()

        for index, (digest, size) in enumerate(
            blobs.items(),
            start=1,
        ):
            console.rule(
                f"[{index}/{len(blobs)}] {human_bytes(size)}"
            )
            console.print(digest)

            try:
                download_blob(
                    client,
                    digest,
                    size,
                    blobs_dir,
                    args.retries,
                    args.lock_timeout,
                    range_size,
                    args.connections,
                )

            except Exception:
                final_path = blobs_dir / digest_filename(digest)
                partial_total = 0

                for path in blobs_dir.glob(
                    final_path.name + ".segment-*.part"
                ):
                    try:
                        partial_total += path.stat().st_size
                    except OSError:
                        pass

                legacy = Path(f"{final_path}.part")

                if legacy.exists():
                    try:
                        partial_total += legacy.stat().st_size
                    except OSError:
                        pass

                if partial_total:
                    console.print(
                        f"  [yellow]Resume data preserved:[/] "
                        f"{human_bytes(partial_total)}"
                    )

                raise

            console.print()

    # Ollama sees the model only after every required blob is valid.
    temp_manifest = Path(str(manifest_path) + ".tmp")

    with temp_manifest.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            manifest,
            f,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    os.replace(temp_manifest, manifest_path)

    display_name = repo.removeprefix("library/")

    console.print("[bold green]Model installed successfully.[/]")
    console.print(f"Manifest: {manifest_path}")
    console.print(
        f"Run with: ollama run {display_name}:{tag}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted.[/] "
            "Run the same command again to resume all segments.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except httpx.HTTPStatusError as exc:
        response = exc.response

        console.print(
            f"\n[bold red]HTTP ERROR:[/] "
            f"{response.status_code} "
            f"{response.reason_phrase} "
            f"for {response.request.url}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except RangeNotSupportedError as exc:
        console.print(
            f"\n[bold red]RANGE ERROR:[/] {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        console.print(
            f"\n[bold red]ERROR:[/] {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
