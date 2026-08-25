# Ollama Model Downloader

A reliable, cross-platform Python downloader for models hosted on the Ollama registry.

It downloads Ollama manifests and blobs directly, supports interrupted-download resume, verifies SHA-256 checksums, and installs models into Ollama's normal local model store.

## Features

- Works on macOS, Linux, and Windows
- No OS-specific shell commands
- Resumable downloads using HTTP `Range`
- Validates `Content-Range` before resuming
- Retries transient network failures automatically
- Retries transient HTTP errors such as `408`, `425`, `429`, `500`, `502`, `503`, and `504`
- Uses randomized exponential backoff
- Verifies completed blobs with SHA-256
- Detects and skips already-valid blobs
- Uses per-blob file locks to prevent concurrent corruption
- Rich terminal progress bars with speed and ETA
- Uses `$OLLAMA_MODELS` when configured
- Otherwise defaults to `~/.ollama/models`
- Supports custom model directories
- Supports custom Ollama-compatible registries

## Requirements

Python 3.9 or newer is recommended.

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

On systems where Python is installed as `python3`:

```bash
python3 -m pip install -r requirements.txt
```

The project uses:

```text
httpx
tenacity
filelock
rich
```

## Files

```text
main.py
requirements.txt
README.md
```

## Basic Usage

Download an Ollama model:

```bash
python main.py qwen3.8:27b
```

On macOS or Linux:

```bash
python3 main.py qwen3.8:27b
```

Other examples:

```bash
python main.py llama3.2:3b
python main.py deepseek-r1:32b
python main.py qwen3.8:27b
```

If no tag is supplied, `latest` is used:

```bash
python main.py llama3.2
```

## Resume Support

Resume is automatic.

If the download is interrupted by:

- network failure
- SSH disconnect
- Ctrl+C
- system reboot
- timeout

run the same command again:

```bash
python3 main.py qwen3.8:27b
```

Incomplete blobs are stored with a `.part` suffix, for example:

```text
sha256-0123456789abcdef.part
```

The downloader checks the partial file size and requests the remaining bytes using HTTP `Range`.

For resumed requests it validates:

- HTTP status `206 Partial Content`
- the returned `Content-Range`

If the server does not return a valid resume response, that individual blob is restarted safely instead of appending potentially invalid data.

After the download completes, the blob is verified using SHA-256 before being moved into its final Ollama blob filename.

## Retry Behavior

By default, retries are unlimited.

The downloader retries common temporary failures including:

```text
408 Request Timeout
425 Too Early
429 Too Many Requests
500 Internal Server Error
502 Bad Gateway
503 Service Unavailable
504 Gateway Timeout
```

It also retries connection errors, read errors, timeouts, protocol errors, and similar transient failures.

Retries use randomized exponential backoff.

To limit retries:

```bash
python3 main.py qwen3.8:27b --retries 10
```

To disable retrying:

```bash
python3 main.py qwen3.8:27b --retries 0
```

## Default Model Directory

The downloader first checks:

```text
OLLAMA_MODELS
```

If `OLLAMA_MODELS` is set, that directory is used.

Otherwise it defaults to:

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

No operating-system-specific logic is required.

## Custom Model Directory

Specify another Ollama model store with:

```bash
python3 main.py qwen3.8:27b --models-dir /mnt/ollama/models
```

Windows example:

```powershell
python main.py qwen3.8:27b --models-dir "D:\ollama\models"
```

## Custom Registry

The default registry is:

```text
https://registry.ollama.ai
```

To use another Ollama-compatible registry:

```bash
python3 main.py mymodel:latest --registry https://example.com
```

## Timeouts

Default connection timeout:

```text
30 seconds
```

Default read inactivity timeout:

```text
120 seconds
```

Default connection-pool timeout:

```text
30 seconds
```

Override them with:

```bash
python3 main.py qwen3.8:27b \
  --connect-timeout 60 \
  --read-timeout 300 \
  --pool-timeout 60
```

## Concurrent Download Protection

Each blob uses a file lock.

If two downloader processes try to download the same blob simultaneously, only one process is allowed to write it.

This helps prevent `.part` file corruption.

The default lock wait is 5 seconds.

Override it with:

```bash
python3 main.py qwen3.8:27b --lock-timeout 30
```

## What Gets Installed

Blobs are written to:

```text
<models-dir>/blobs/
```

Manifests are written to:

```text
<models-dir>/manifests/<registry>/<repository>/<tag>
```

Example:

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

The manifest is installed only after all required blobs have downloaded and passed verification.

## After Download

Check that Ollama sees the model:

```bash
ollama list
```

Run the model:

```bash
ollama run qwen3.8:27b
```

## Integrity Verification

For every blob, the downloader:

1. Checks whether a local final blob already exists.
2. Checks its expected size.
3. Verifies its SHA-256 checksum.
4. Resumes a partial `.part` file when possible.
5. Validates HTTP resume responses.
6. Verifies the completed `.part` file.
7. Atomically moves it into the final Ollama blob filename.

A corrupted completed blob is not accepted.

## Command-Line Help

```bash
python3 main.py --help
```

## Example: qwen3.8:27b

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download:

```bash
python3 main.py qwen3.8:27b
```

If interrupted, run the same command again:

```bash
python3 main.py qwen3.8:27b
```

Then run:

```bash
ollama run qwen3.8:27b
```

## Notes

- Resume depends on the registry/blob server supporting HTTP byte-range requests.
- Large models require substantial free disk space.
- The downloader does not start or stop the Ollama server.
- Ollama and this downloader should point to the same model directory.
- If you use `OLLAMA_MODELS`, make sure Ollama itself uses the same value.
- Models remain subject to their own licenses and usage terms.
