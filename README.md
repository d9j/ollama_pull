# Ollama Model Downloader

A cross-platform Python downloader for models hosted on the Ollama registry.

It downloads Ollama manifests and blobs directly, supports interrupted-download resume, verifies SHA-256 checksums, and installs models into Ollama's normal local model store.

## Features

- Works on macOS, Linux, and Windows
- No OS-specific shell commands
- Uses a small reliability-focused dependency set
- HTTP connection pooling and streaming via `httpx`
- Retry/backoff with jitter via `tenacity`
- Cross-process blob locking via `filelock`
- Download progress display via `rich`
- Supports normal Ollama model references such as `qwen3.8:27b`
- Automatically downloads the model manifest and all required blobs
- Resumes interrupted blob downloads using HTTP `Range` requests
- Automatically retries interrupted downloads
- Verifies every completed blob with SHA-256
- Detects and skips already-complete valid blobs
- Uses `$OLLAMA_MODELS` when set
- Otherwise defaults to `~/.ollama/models`
- Supports a custom model directory
- Supports a custom Ollama-compatible registry

## Requirements

- Python 3.9 or newer recommended
- Network access to the Ollama registry
- Ollama installed if you want to run the downloaded model afterward

Python dependencies are listed in `requirements.txt`:

```text
httpx
tenacity
filelock
rich
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

On systems where the Python executable is named `python3`:

```bash
python3 -m pip install -r requirements.txt
```

## Download

Save the script as:

```text
ollama_download.py
```

## Basic Usage

Download a model:

```bash
python ollama_download.py qwen3.8:27b
```

On macOS or Linux you may use:

```bash
python3 ollama_download.py qwen3.8:27b
```

Other examples:

```bash
python ollama_download.py llama3.2:3b
python ollama_download.py deepseek-r1:32b
python ollama_download.py qwen3.8:27b
```

If no tag is supplied, `latest` is used:

```bash
python ollama_download.py llama3.2
```

## Resume Support

Resume is automatic.

If the download is interrupted by:

- network failure
- SSH disconnect
- Ctrl+C
- system reboot
- timeout

just run the same command again:

```bash
python ollama_download.py qwen3.8:27b
```

Incomplete blobs are stored with a `.part` suffix, for example:

```text
sha256-0123456789abcdef.part
```

On the next run, the downloader checks the existing partial size and requests the remaining bytes using an HTTP `Range` request.

After the blob finishes downloading, the script verifies its SHA-256 checksum before installing it.

For resumed requests, the downloader validates both:

- HTTP status `206 Partial Content`
- the returned `Content-Range`

If the server does not honor the resume request correctly, that individual blob is restarted from the beginning rather than appending unsafe data.

A per-blob file lock also prevents two downloader processes from writing the same partial blob simultaneously.

## Reliability

The downloader uses:

- `httpx.Client` for persistent HTTP connections, streaming, redirects, proxies, and separate connection/read/pool timeouts
- `tenacity` for retries with randomized exponential backoff
- `filelock` to protect blobs from concurrent writers
- `rich` for byte-aware progress reporting

Transient HTTP statuses are retried automatically, including:

```text
408
425
429
500
502
503
504
```

Network interruptions such as connection errors, read errors, and timeouts are also retried.

By default retries are unlimited. Use `--retries` to limit them.

## Default Model Directory

The downloader first checks the `OLLAMA_MODELS` environment variable.

If it is defined, that directory is used.

Otherwise the script uses:

```text
~/.ollama/models
```

Examples:

### macOS

```text
/Users/alexdev/.ollama/models
```

### Linux

```text
/home/alexdev/.ollama/models
```

### Windows

```text
C:\Users\alexdev\.ollama\models
```

## Custom Model Directory

You can override the destination directory:

```bash
python ollama_download.py qwen3.8:27b --models-dir /mnt/ollama/models
```

Windows example:

```powershell
python ollama_download.py qwen3.8:27b --models-dir "D:\ollama\models"
```

## Custom Registry

The default registry is:

```text
https://registry.ollama.ai
```

To use another compatible registry:

```bash
python ollama_download.py mymodel:latest --registry https://example.com
```

## Retry Behavior

By default, interrupted blob downloads retry indefinitely.

To limit the number of retries per blob:

```bash
python ollama_download.py qwen3.8:27b --retries 10
```

Use:

```text
--retries -1
```

for unlimited retries.

## HTTP Timeout

The default HTTP timeout is 60 seconds.

Override it with:

```bash
python ollama_download.py qwen3.8:27b --timeout 120
```

## Command-Line Help

```bash
python ollama_download.py --help
```

## What Gets Installed

The downloader writes blobs into:

```text
<models-dir>/blobs/
```

and manifests into:

```text
<models-dir>/manifests/<registry>/<repository>/<tag>
```

For example:

```text
~/.ollama/models/
├── blobs/
│   ├── sha256-...
│   └── sha256-...
└── manifests/
    └── registry.ollama.ai/
        └── library/
            └── qwen3.8/
                └── 27b
```

This is the same general layout Ollama uses for locally stored registry models.

## After Download

Check that Ollama sees the model:

```bash
ollama list
```

Run it:

```bash
ollama run qwen3.8:27b
```

## Integrity Verification

For every blob, the downloader:

1. Checks whether a complete local blob already exists.
2. Verifies its expected file size.
3. Verifies the SHA-256 checksum.
4. Downloads or resumes the blob when required.
5. Verifies the completed `.part` file.
6. Renames it to the final Ollama blob filename only after successful verification.

The model manifest is installed only after all required blobs have completed successfully.

## Interrupted Downloads

If you intentionally stop the downloader with Ctrl+C:

```text
Interrupted. Run the same command again to resume.
```

Simply rerun the original command.

Do not delete the `.part` files if you want to resume.

## Notes

- Resume requires the remote registry/blob server to support HTTP byte-range requests.
- Large models may require substantial free disk space.
- The downloader does not start or stop the Ollama server.
- Ollama and this downloader should point to the same model directory.
- If you use `OLLAMA_MODELS`, make sure Ollama itself is configured with the same value.

## Example: qwen3.8:27b

Download:

```bash
python ollama_download.py qwen3.8:27b
```

If interrupted:

```bash
python ollama_download.py qwen3.8:27b
```

Then run:

```bash
ollama run qwen3.8:27b
```

## License

Use and modify the script as needed. Models downloaded with it remain subject to their own licenses and usage terms.
