#!/usr/bin/env python3
"""
Ollama Model Downloader v3

Cross-platform, resumable downloader for Ollama-compatible registries.

Reliability changes in v3:
  - Large blobs are downloaded with bounded HTTP Range requests instead of
    one long-lived response.
  - If a request disconnects halfway through, the next retry resumes from
    the current .part file size.
  - Short read-inactivity timeout by default.
  - Finite retry count by default so a dead endpoint does not retry forever.
  - Validates HTTP 206 and Content-Range for every ranged response.
  - Preserves partial files on failure so rerunning the command resumes.

Dependencies: httpx, tenacity, filelock, rich
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
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
STREAM_CHUNK_SIZE = 1024 * 1024
DEFAULT_RANGE_SIZE_MIB = 32
DEFAULT_RETRIES = 10
DEFAULT_CONNECT_TIMEOUT = 20.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 20.0
USER_AGENT = "ollama-cross-platform-downloader/3.0"

RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
_AUTH_RE = re.compile(r'(\w+)="([^"]*)"')
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.I)

console = Console()


class RetryableHTTPStatus(httpx.HTTPStatusError):
    """Temporary HTTP status that should be retried."""


class RetryableRangeError(OSError):
    """Temporary malformed/truncated Range response."""


class RangeNotSupportedError(RuntimeError):
    """Range semantics were ignored, making safe resume impossible."""


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
    return Path(configured).expanduser() if configured else Path.home() / ".ollama" / "models"


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

    return (name if "/" in name else f"library/{name}"), tag


def digest_filename(digest: str) -> str:
    algo, sep, value = digest.partition(":")
    if not sep or not algo or not value:
        raise ValueError(f"invalid digest: {digest}")
    return f"{algo}-{value}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class RegistryClient:
    def __init__(
        self,
        base_url: str,
        repo: str,
        *,
        connect_timeout: float,
        read_timeout: float,
        pool_timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.repo = repo
        self.token: Optional[str] = None

        self.client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=30.0,
                pool=pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=4,
                keepalive_expiry=20.0,
            ),
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
            },
        )

    def __enter__(self) -> "RegistryClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.client.close()

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get_bearer_token(self, challenge: str) -> str:
        params = dict(_AUTH_RE.findall(challenge))
        realm = params.get("realm")
        if not realm:
            raise RuntimeError("registry requested Bearer auth but supplied no realm")

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

    def _send(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        merged = self._auth_headers()
        if headers:
            merged.update(headers)

        if stream:
            request = self.client.build_request(method, url, headers=merged)
            return self.client.send(request, stream=True)
        return self.client.request(method, url, headers=merged)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        response = self._send(method, url, headers=headers, stream=stream)

        # Refresh an absent or expired Bearer token once.
        if response.status_code == 401:
            challenge = response.headers.get("WWW-Authenticate", "")
            response.close()
            if challenge.lower().startswith("bearer "):
                self.token = None
                self.token = self._get_bearer_token(challenge)
                response = self._send(method, url, headers=headers, stream=stream)

        if response.status_code in RETRYABLE_HTTP_STATUSES:
            status = response.status_code
            reason = response.reason_phrase
            request = response.request
            response.close()
            raise RetryableHTTPStatus(
                f"temporary HTTP {status} {reason}",
                request=request,
                response=httpx.Response(status, request=request, reason_phrase=reason),
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

    def open_blob_range(self, digest: str, start: int, end: int) -> httpx.Response:
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
    stop_rule = stop_never if retries < 0 else stop_after_attempt(retries + 1)
    return Retrying(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_rule,
        before_sleep=retry_message,
        reraise=True,
    )


def fetch_manifest_with_retry(client: RegistryClient, tag: str, retries: int) -> dict:
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


def valid_existing_blob(
    final_path: Path,
    part_path: Path,
    expected_size: int,
    expected_hash: str,
) -> bool:
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

    if local_size < expected_size and not part_path.exists():
        console.print(
            f"  Converting incomplete blob to .part "
            f"({human_bytes(local_size)} already present)"
        )
        final_path.replace(part_path)
    else:
        final_path.unlink(missing_ok=True)
    return False


def parse_content_range(value: str) -> Tuple[int, int, Optional[int]]:
    match = _CONTENT_RANGE_RE.match(value.strip())
    if not match:
        raise RetryableRangeError(f"invalid or missing Content-Range: {value!r}")
    start = int(match.group(1))
    end = int(match.group(2))
    total_text = match.group(3)
    total = None if total_text == "*" else int(total_text)
    return start, end, total


def download_next_range_once(
    client: RegistryClient,
    digest: str,
    expected_size: int,
    part_path: Path,
    range_size: int,
    progress: Progress,
    task_id: int,
) -> int:
    """
    Download one bounded range.

    The start offset is recalculated from the current .part size on every
    retry. If the connection drops after writing some bytes, the retry starts
    from those newly written bytes instead of restarting the range.
    """
    start = part_path.stat().st_size if part_path.exists() else 0
    if start >= expected_size:
        return 0

    requested_end = min(expected_size - 1, start + range_size - 1)
    response = client.open_blob_range(digest, start, requested_end)

    try:
        if response.status_code == 200:
            # Some servers ignore Range for tiny files. Accept this only when
            # byte 0 was requested and the entire blob fits in one window.
            content_length_text = response.headers.get("Content-Length")
            try:
                content_length = int(content_length_text) if content_length_text else None
            except ValueError:
                content_length = None

            if start == 0 and expected_size <= range_size and content_length == expected_size:
                response_start = 0
                response_end = expected_size - 1
            else:
                raise RangeNotSupportedError(
                    "registry/CDN ignored the HTTP Range request; safe resume is not possible"
                )

        elif response.status_code == 206:
            response_start, response_end, response_total = parse_content_range(
                response.headers.get("Content-Range", "")
            )

            if response_start != start:
                raise RetryableRangeError(
                    f"wrong Content-Range start: {response_start} != {start}"
                )
            if response_end > requested_end:
                raise RetryableRangeError(
                    f"server returned beyond requested range: {response_end} > {requested_end}"
                )
            if response_end < response_start:
                raise RetryableRangeError("invalid Content-Range end")
            if response_total is not None and response_total != expected_size:
                raise RetryableRangeError(
                    f"wrong Content-Range total: {response_total} != {expected_size}"
                )
        else:
            raise RetryableRangeError(
                f"unexpected HTTP status for Range request: {response.status_code}"
            )

        expected_bytes = response_end - response_start + 1
        written = 0

        with part_path.open("ab") as out:
            for chunk in response.iter_raw(STREAM_CHUNK_SIZE):
                if not chunk:
                    continue
                remaining = expected_bytes - written
                if remaining <= 0 or len(chunk) > remaining:
                    raise RetryableRangeError(
                        "server sent more bytes than declared by Content-Range"
                    )
                out.write(chunk)
                written += len(chunk)
                progress.advance(task_id, len(chunk))
            out.flush()

        if written != expected_bytes:
            raise RetryableRangeError(
                f"truncated range: received {written} bytes, expected {expected_bytes}"
            )

        actual_size = part_path.stat().st_size
        expected_after = start + expected_bytes
        if actual_size != expected_after:
            raise RetryableRangeError(
                f"partial size mismatch: {actual_size} != {expected_after}"
            )

        return written
    finally:
        response.close()


def download_blob(
    client: RegistryClient,
    digest: str,
    expected_size: int,
    blobs_dir: Path,
    retries: int,
    lock_timeout: float,
    range_size: int,
) -> None:
    algorithm, _, expected_hash = digest.partition(":")
    if algorithm != "sha256":
        raise RuntimeError(f"unsupported digest algorithm: {algorithm}")

    final_path = blobs_dir / digest_filename(digest)
    part_path = Path(str(final_path) + ".part")
    lock_path = Path(str(final_path) + ".lock")

    try:
        with FileLock(str(lock_path), timeout=lock_timeout):
            if valid_existing_blob(final_path, part_path, expected_size, expected_hash):
                return

            if part_path.exists():
                current_size = part_path.stat().st_size
                if current_size > expected_size:
                    console.print("  [yellow]Partial blob is too large; restarting it.[/]")
                    part_path.unlink()
                    current_size = 0
                elif current_size:
                    console.print(
                        f"  [cyan]Resuming at {human_bytes(current_size)} "
                        f"of {human_bytes(expected_size)}[/]"
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

            with progress:
                task_id = progress.add_task(
                    digest[:20] + "…",
                    total=expected_size,
                    completed=part_path.stat().st_size if part_path.exists() else 0,
                )

                while True:
                    current_size = part_path.stat().st_size if part_path.exists() else 0
                    if current_size >= expected_size:
                        break

                    downloaded: Optional[int] = None
                    for attempt in make_retryer(retries):
                        with attempt:
                            downloaded = download_next_range_once(
                                client,
                                digest,
                                expected_size,
                                part_path,
                                range_size,
                                progress,
                                task_id,
                            )

                    if downloaded is None or downloaded <= 0:
                        raise RuntimeError("range downloader made no forward progress")

            actual_size = part_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"blob size mismatch before verification: {actual_size} != {expected_size}"
                )

            console.print("  Verifying SHA-256...")
            actual_hash = sha256_file(part_path)
            if actual_hash != expected_hash:
                corrupt_path = Path(str(part_path) + ".corrupt")
                try:
                    corrupt_path.unlink(missing_ok=True)
                    os.replace(part_path, corrupt_path)
                except OSError:
                    part_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
                )

            os.replace(part_path, final_path)
            console.print("  [bold green]OK[/]")

    except FileLockTimeout as exc:
        raise RuntimeError(
            f"could not obtain lock for {final_path.name}; another downloader may be active"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reliable cross-platform Ollama registry model downloader."
    )
    parser.add_argument("model", help="Model reference, e.g. qwen3.8:27b")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_models_dir(),
        help="Ollama models directory ($OLLAMA_MODELS or ~/.ollama/models)",
    )
    parser.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help=f"Registry URL (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=(
            "Retries per failed Range request. "
            f"Default: {DEFAULT_RETRIES}. Use -1 for unlimited."
        ),
    )
    parser.add_argument(
        "--range-size-mib",
        type=int,
        default=DEFAULT_RANGE_SIZE_MIB,
        help=(
            "Maximum bytes requested per HTTP Range request in MiB "
            f"(default: {DEFAULT_RANGE_SIZE_MIB})"
        ),
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        help=f"Connection timeout in seconds (default: {DEFAULT_CONNECT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=DEFAULT_READ_TIMEOUT,
        help=(
            "Seconds with no incoming data before retry "
            f"(default: {DEFAULT_READ_TIMEOUT:g})"
        ),
    )
    parser.add_argument(
        "--pool-timeout",
        type=float,
        default=DEFAULT_POOL_TIMEOUT,
        help=f"Pool timeout in seconds (default: {DEFAULT_POOL_TIMEOUT:g})",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a per-blob lock (default: 5)",
    )

    args = parser.parse_args()

    if args.retries < -1:
        parser.error("--retries must be -1 or greater")
    if not 1 <= args.range_size_mib <= 1024:
        parser.error("--range-size-mib must be between 1 and 1024")
    for name in ("connect_timeout", "read_timeout", "pool_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
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
    console.print(f"[bold]Range size:[/]   {args.range_size_mib} MiB")
    console.print(f"[bold]Read timeout:[/] {args.read_timeout:g}s inactivity")
    console.print(
        f"[bold]Retries:[/]      "
        + ("unlimited" if args.retries < 0 else str(args.retries))
    )
    console.print()

    with RegistryClient(
        registry,
        repo,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        pool_timeout=args.pool_timeout,
    ) as client:
        console.print("[bold]Fetching manifest...[/]")
        manifest = fetch_manifest_with_retry(client, tag, args.retries)

        blobs: Dict[str, int] = {}
        for obj in manifest_objects(manifest):
            digest = obj.get("digest")
            if not digest:
                continue
            size = int(obj.get("size") or 0)
            if size <= 0:
                raise RuntimeError(f"invalid size for {digest}: {size}")
            blobs.setdefault(digest, size)

        if not blobs:
            raise RuntimeError("manifest contains no downloadable blobs")

        console.print(f"[bold]Blobs:[/]        {len(blobs)}")
        console.print(f"[bold]Total size:[/]   {human_bytes(sum(blobs.values()))}")
        console.print()

        for index, (digest, size) in enumerate(blobs.items(), start=1):
            console.rule(f"[{index}/{len(blobs)}] {human_bytes(size)}")
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
                )
            except Exception:
                partial = blobs_dir / (digest_filename(digest) + ".part")
                if partial.exists():
                    console.print(
                        f"  [yellow]Partial download preserved:[/] "
                        f"{human_bytes(partial.stat().st_size)}"
                    )
                raise
            console.print()

    # Ollama sees the model only after every required blob is complete.
    temp_manifest = Path(str(manifest_path) + ".tmp")
    with temp_manifest.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(temp_manifest, manifest_path)

    display_name = repo.removeprefix("library/")
    console.print("[bold green]Model installed successfully.[/]")
    console.print(f"Manifest: {manifest_path}")
    console.print(f"Run with: ollama run {display_name}:{tag}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted.[/] Run the same command again to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except RangeNotSupportedError as exc:
        console.print(f"\n[bold red]RANGE ERROR:[/] {exc}", file=sys.stderr)
        raise SystemExit(1)
    except httpx.HTTPStatusError as exc:
        response = exc.response
        console.print(
            f"\n[bold red]HTTP ERROR:[/] {response.status_code} "
            f"{response.reason_phrase} for {response.request.url}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"\n[bold red]ERROR:[/] {exc}", file=sys.stderr)
        raise SystemExit(1)
