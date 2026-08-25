#!/usr/bin/env python3
"""
Cross-platform Ollama registry model downloader.

Downloads a model from an OCI/Docker Registry-compatible Ollama registry,
resumes interrupted blobs, verifies SHA-256, and installs the manifest in
Ollama's normal local model store.

Examples:
  python ollama_download.py qwen3.8:27b
  python ollama_download.py llama3.2:3b
  python ollama_download.py myorg/mymodel:latest
  python ollama_download.py qwen3.8:27b --models-dir D:/ollama/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


DEFAULT_REGISTRY = "https://registry.ollama.ai"
CHUNK_SIZE = 4 * 1024 * 1024
USER_AGENT = "ollama-cross-platform-downloader/1.0"

_AUTH_RE = re.compile(r'(\w+)="([^"]*)"')


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def default_models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".ollama" / "models"


def parse_model_ref(ref: str) -> Tuple[str, str]:
    ref = ref.strip().strip("/")
    if not ref:
        raise ValueError("empty model name")

    for prefix in (
        "https://registry.ollama.ai/",
        "http://registry.ollama.ai/",
        "registry.ollama.ai/",
    ):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            break

    last = ref.rsplit("/", 1)[-1]
    if ":" in last:
        name, tag = ref.rsplit(":", 1)
    else:
        name, tag = ref, "latest"

    if not name or not tag:
        raise ValueError(f"invalid model reference: {ref!r}")

    repo = name if "/" in name else f"library/{name}"
    return repo, tag


def digest_filename(digest: str) -> str:
    algo, sep, value = digest.partition(":")
    if sep != ":" or not algo or not value:
        raise ValueError(f"invalid digest: {digest}")
    return f"{algo}-{value}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class RegistryClient:
    def __init__(self, base_url: str, repo: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.repo = repo
        self.timeout = timeout
        self.token: Optional[str] = None

    def _request(self, url: str, headers: Optional[Dict[str, str]] = None):
        hdrs = {
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        }
        if headers:
            hdrs.update(headers)
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"

        req = Request(url, headers=hdrs)
        try:
            return urlopen(req, timeout=self.timeout)
        except HTTPError as exc:
            if exc.code == 401 and not self.token:
                challenge = exc.headers.get("WWW-Authenticate", "")
                if challenge.lower().startswith("bearer "):
                    self.token = self._get_bearer_token(challenge)
                    hdrs["Authorization"] = f"Bearer {self.token}"
                    return urlopen(Request(url, headers=hdrs), timeout=self.timeout)
            raise

    def _get_bearer_token(self, challenge: str) -> str:
        params = dict(_AUTH_RE.findall(challenge))
        realm = params.get("realm")
        if not realm:
            raise RuntimeError("registry requested Bearer auth but provided no realm")

        query = {}
        if params.get("service"):
            query["service"] = params["service"]
        query["scope"] = params.get("scope") or f"repository:{self.repo}:pull"

        sep = "&" if "?" in realm else "?"
        token_url = realm + sep + urlencode(query)

        req = Request(token_url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)

        token = payload.get("token") or payload.get("access_token")
        if not token:
            raise RuntimeError("registry authentication response contained no token")
        return token

    def manifest(self, tag: str) -> dict:
        url = f"{self.base_url}/v2/{self.repo}/manifests/{quote(tag, safe='')}"
        headers = {
            "Accept": (
                "application/vnd.docker.distribution.manifest.v2+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/json"
            )
        }
        with self._request(url, headers) as resp:
            return json.load(resp)

    def blob_url(self, digest: str) -> str:
        return f"{self.base_url}/v2/{self.repo}/blobs/{quote(digest, safe=':')}"

    def open_blob(self, digest: str, offset: int = 0):
        headers = {}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        return self._request(self.blob_url(digest), headers)


def download_blob(
    client: RegistryClient,
    digest: str,
    expected_size: int,
    blobs_dir: Path,
    retries: int,
) -> None:
    algo, _, expected_hash = digest.partition(":")
    if algo != "sha256":
        raise RuntimeError(f"unsupported digest algorithm: {algo}")

    final_path = blobs_dir / digest_filename(digest)
    part_path = Path(str(final_path) + ".part")

    if final_path.exists():
        size = final_path.stat().st_size

        if size == expected_size:
            print(f"  existing {final_path.name}; verifying SHA-256...")
            if sha256_file(final_path) == expected_hash:
                print("  OK; skipped")
                return

            print("  checksum mismatch; redownloading")
            final_path.unlink()

        elif size < expected_size and not part_path.exists():
            final_path.replace(part_path)

        else:
            final_path.unlink()

    if part_path.exists() and part_path.stat().st_size > expected_size:
        part_path.unlink()

    attempt = 0

    while True:
        attempt += 1

        offset = part_path.stat().st_size if part_path.exists() else 0
        mode = "ab" if offset else "wb"

        try:
            with client.open_blob(digest, offset) as resp:
                status = getattr(resp, "status", resp.getcode())

                if offset and status != 206:
                    print("  server did not honor resume request; restarting this blob")
                    offset = 0
                    mode = "wb"

                downloaded = offset
                last_report = time.monotonic()
                last_bytes = downloaded

                with part_path.open(mode) as out:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        out.write(chunk)
                        downloaded += len(chunk)

                        now = time.monotonic()

                        if now - last_report >= 1.0:
                            interval = max(now - last_report, 1e-6)
                            speed = (downloaded - last_bytes) / interval
                            pct = (
                                downloaded / expected_size * 100.0
                                if expected_size
                                else 0.0
                            )

                            print(
                                f"\r  {pct:6.2f}%  "
                                f"{human_bytes(downloaded)} / "
                                f"{human_bytes(expected_size)}  "
                                f"{human_bytes(int(speed))}/s",
                                end="",
                                flush=True,
                            )

                            last_report = now
                            last_bytes = downloaded

            print()

            actual_size = part_path.stat().st_size

            if actual_size != expected_size:
                raise IOError(
                    f"incomplete blob: got {actual_size} bytes, "
                    f"expected {expected_size}"
                )

            print("  verifying SHA-256...")

            actual_hash = sha256_file(part_path)

            if actual_hash != expected_hash:
                part_path.unlink(missing_ok=True)
                raise IOError(
                    f"SHA-256 mismatch: got {actual_hash}, "
                    f"expected {expected_hash}"
                )

            os.replace(part_path, final_path)
            print("  OK")
            return

        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if retries >= 0 and attempt > retries:
                raise RuntimeError(
                    f"failed downloading {digest}: {exc}"
                ) from exc

            delay = min(30, 2 ** min(attempt, 5))
            current = part_path.stat().st_size if part_path.exists() else 0

            print(
                f"\n  download interrupted at {human_bytes(current)}: {exc}\n"
                f"  retrying in {delay}s (resume enabled)..."
            )

            time.sleep(delay)


def manifest_objects(manifest: dict) -> Iterable[dict]:
    config = manifest.get("config")

    if isinstance(config, dict):
        yield config

    layers = manifest.get("layers")

    if isinstance(layers, list):
        yield from layers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform resumable Ollama registry model downloader."
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
            "(default: $OLLAMA_MODELS or ~/.ollama/models)"
        ),
    )

    parser.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help=f"Registry base URL (default: {DEFAULT_REGISTRY})",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=-1,
        help="Retries per interrupted blob; -1 = unlimited (default)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    try:
        repo, tag = parse_model_ref(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    models_dir = args.models_dir.expanduser().resolve()
    blobs_dir = models_dir / "blobs"

    registry_url = args.registry.rstrip("/")
    parsed = urlparse(registry_url)

    if not parsed.scheme or not parsed.netloc:
        parser.error(
            "--registry must be a full URL such as "
            "https://registry.ollama.ai"
        )

    registry_host = parsed.netloc

    manifest_path = (
        models_dir
        / "manifests"
        / registry_host
        / Path(*repo.split("/"))
        / tag
    )

    blobs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    client = RegistryClient(
        registry_url,
        repo,
        timeout=args.timeout,
    )

    print(f"Model      : {repo}:{tag}")
    print(f"Registry   : {registry_url}")
    print(f"Models dir : {models_dir}")
    print()

    print("Fetching manifest...")
    manifest = client.manifest(tag)

    objects = list(manifest_objects(manifest))

    if not objects:
        raise RuntimeError("manifest contains no config/layers")

    unique: Dict[str, int] = {}

    for obj in objects:
        digest = obj.get("digest")

        if not digest:
            continue

        size = int(obj.get("size") or 0)

        if digest not in unique:
            unique[digest] = size

    if not unique:
        raise RuntimeError("manifest contains no downloadable digests")

    total = sum(unique.values())

    print(f"Blobs      : {len(unique)}")
    print(f"Total size : {human_bytes(total)}")
    print()

    for index, (digest, size) in enumerate(unique.items(), 1):
        print(
            f"[{index}/{len(unique)}] "
            f"{digest}  ({human_bytes(size)})"
        )

        download_blob(
            client,
            digest,
            size,
            blobs_dir,
            args.retries,
        )

        print()

    # Only install the manifest after every blob has been
    # completely downloaded and checksum-verified.
    tmp_manifest = Path(str(manifest_path) + ".tmp")

    with tmp_manifest.open(
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

    os.replace(tmp_manifest, manifest_path)

    display_name = (
        repo[len("library/"):]
        if repo.startswith("library/")
        else repo
    )

    print("Installed successfully.")
    print(f"Manifest: {manifest_path}")
    print()
    print(f"Run with: ollama run {display_name}:{tag}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Run the same command again to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
