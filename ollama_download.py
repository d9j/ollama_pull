#!/usr/bin/env python3
"""
Ollama Model Downloader v2

Cross-platform, resumable downloader for Ollama-compatible registries.

Reliability stack:
  - httpx    : streaming HTTP, connection pooling, redirects, timeouts
  - tenacity : retry/backoff with jitter
  - filelock : prevents concurrent writers corrupting the same blob
  - rich     : progress bars and terminal status

Examples:
  python ollama_download_v2.py qwen3.8:27b
  python ollama_download_v2.py llama3.2:3b
  python ollama_download_v2.py qwen3.8:27b --models-dir D:/ollama/models
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
CHUNK_SIZE = 4 * 1024 * 1024
USER_AGENT = "ollama-cross-platform-downloader/2.0"

RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
_AUTH_RE = re.compile(r'(\w+)="([^"]*)"')

console = Console()


class RetryableHTTPStatus(httpx.HTTPStatusError):
    """Temporary HTTP error that should be retried."""


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def default_models_dir() -> Path:
    """Use OLLAMA_MODELS when set; otherwise ~/.ollama/models."""
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
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
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

    def _authorization_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get_bearer_token(self, challenge: str) -> str:
        params = dict(_AUTH_RE.findall(challenge))
        realm = params.get("realm")
        if not realm:
            raise RuntimeError("Bearer authentication challenge has no realm")

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

    def request(
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
            request = self.client.build_request(method, url, headers=merged)
            response = self.client.send(request, stream=True)
        else:
            response = self.client.request(method, url, headers=merged)

        if response.status_code == 401 and not self.token:
            challenge = response.headers.get("WWW-Authenticate", "")
            response.close()

            if challenge.lower().startswith("bearer "):
                self.token = self._get_bearer_token(challenge)
                merged = self._authorization_headers()
                if headers:
                    merged.update(headers)

                if stream:
                    request = self.client.build_request(method, url, headers=merged)
                    response = self.client.send(request, stream=True)
                else:
                    response = self.client.request(method, url, headers=merged)

        if response.status_code in RETRYABLE_HTTP_STATUSES:
            status = response.status_code
            reason = response.reason_phrase
            request = response.request
            response.close()
            raise RetryableHTTPStatus(
                f"temporary HTTP {status} {reason}",
                request=request,
                response=httpx.Response(status, request=request),
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

    def open_blob(self, digest: str, offset: int = 0) -> httpx.Response:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        return self.request(
            "GET",
            f"{self.base_url}/v2/{self.repo}/blobs/{digest}",
            headers=headers,
            stream=True,
        )


RETRYABLE_EXCEPTIONS = (
    RetryableHTTPStatus,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.WriteError,
    OSError,
)


def on_retry(retry_state: RetryCallState) -> None:
    wait = retry_state.next_action.sleep if retry_state.next_action else 0.0
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    console.print(
        f"  [yellow]Retrying in {wait:.1f}s[/]"
        + (f" — {exc}" if exc else "")
    )


def make_retryer(retries: int) -> Retrying:
    stop_rule = stop_never if retries < 0 else stop_after_attempt(retries + 1)

    return Retrying(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_rule,
        before_sleep=on_retry,
        reraise=True,
    )


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

    # Preserve a short/incomplete file as a resumable .part file.
    if local_size < expected_size and not part_path.exists():
        final_path.replace(part_path)
    else:
        final_path.unlink(missing_ok=True)

    return False


def download_blob_attempt(
    client: RegistryClient,
    digest: str,
    expected_size: int,
    part_path: Path,
    progress: Progress,
    task_id: int,
) -> None:
    offset = part_path.stat().st_size if part_path.exists() else 0
    response = client.open_blob(digest, offset)

    try:
        if offset:
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {offset}-"

            if response.status_code != 206 or not content_range.startswith(expected_prefix):
                console.print(
                    "  [yellow]Registry did not return a valid Range response; "
                    "restarting this blob safely.[/]"
                )
                response.close()
                part_path.unlink(missing_ok=True)
                offset = 0
                response = client.open_blob(digest, 0)

        mode = "ab" if offset else "wb"
        progress.update(task_id, completed=offset)

        with part_path.open(mode) as out:
            # Raw bytes are important: no transparent content decoding.
            for chunk in response.iter_raw(CHUNK_SIZE):
                if chunk:
                    out.write(chunk)
                    progress.advance(task_id, len(chunk))

        actual_size = part_path.stat().st_size
        if actual_size != expected_size:
            raise OSError(
                f"incomplete blob: got {actual_size} bytes, expected {expected_size}"
            )
    finally:
        response.close()


def download_blob(
    client: RegistryClient,
    digest: str,
    expected_size: int,
    blobs_dir: Path,
    retries: int,
    lock_timeout: float,
) -> None:
    algorithm, _, expected_hash = digest.partition(":")
    if algorithm != "sha256":
        raise RuntimeError(f"unsupported digest algorithm: {algorithm}")

    final_path = blobs_dir / digest_filename(digest)
    part_path = Path(str(final_path) + ".part")
    lock_path = Path(str(final_path) + ".lock")

    try:
        with FileLock(str(lock_path), timeout=lock_timeout):
            if valid_existing_blob(
                final_path,
                part_path,
                expected_size,
                expected_hash,
            ):
                return

            if part_path.exists() and part_path.stat().st_size > expected_size:
                console.print(
                    "  [yellow]Partial blob is larger than expected; restarting it.[/]"
                )
                part_path.unlink()

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
                task = progress.add_task(
                    digest[:20] + "…",
                    total=expected_size,
                    completed=part_path.stat().st_size if part_path.exists() else 0,
                )

                for attempt in make_retryer(retries):
                    with attempt:
                        download_blob_attempt(
                            client,
                            digest,
                            expected_size,
                            part_path,
                            progress,
                            task,
                        )

            console.print("  Verifying SHA-256...")
            actual_hash = sha256_file(part_path)

            if actual_hash != expected_hash:
                part_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
                )

            os.replace(part_path, final_path)
            console.print("  [bold green]OK[/]")

    except FileLockTimeout as exc:
        raise RuntimeError(
            f"could not obtain lock for {final_path.name}; "
            "another downloader may already be downloading this blob"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reliable cross-platform Ollama registry model downloader."
    )
    parser.add_argument("model", help="Model, e.g. qwen3.8:27b")
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
        default=-1,
        help="Retries per interrupted blob; -1 = unlimited (default)",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=120.0)
    parser.add_argument("--pool-timeout", type=float, default=30.0)
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a per-blob file lock",
    )

    args = parser.parse_args()

    for name in ("connect_timeout", "read_timeout", "pool_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")

    if args.lock_timeout < 0:
        parser.error("--lock-timeout must be >= 0")

    repo, tag = parse_model_ref(args.model)
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

    console.print(f"[bold]Model:[/]      {repo}:{tag}")
    console.print(f"[bold]Registry:[/]   {registry}")
    console.print(f"[bold]Models dir:[/] {models_dir}")
    console.print()

    with RegistryClient(
        registry,
        repo,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        pool_timeout=args.pool_timeout,
    ) as client:
        console.print("[bold]Fetching manifest...[/]")
        manifest = client.get_manifest(tag)

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

        console.print(f"[bold]Blobs:[/]      {len(blobs)}")
        console.print(f"[bold]Total size:[/] {human_bytes(sum(blobs.values()))}")
        console.print()

        for index, (digest, size) in enumerate(blobs.items(), start=1):
            console.rule(f"[{index}/{len(blobs)}] {human_bytes(size)}")
            console.print(digest)

            download_blob(
                client,
                digest,
                size,
                blobs_dir,
                args.retries,
                args.lock_timeout,
            )
            console.print()

    # Manifest is made visible to Ollama only after every blob is valid.
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
    except httpx.HTTPStatusError as exc:
        response = exc.response
        console.print(
            f"\n[bold red]HTTP ERROR:[/] {response.status_code} "
            f"for {response.request.url}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"\n[bold red]ERROR:[/] {exc}", file=sys.stderr)
        raise SystemExit(1)
