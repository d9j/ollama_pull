# Ollama Model Downloader

Cross-platform resumable **multi-connection** downloader for Ollama registry models.

By default, the downloader uses `aria2c` for blob transfers when it is
available on `PATH`. It automatically falls back to the built-in HTTPX
downloader when aria2c is unavailable or cannot start a transfer.

## v4: aria2-style parallel downloads

v4 can use several HTTP Range connections for a single large blob.

Example:

```text
540 MiB blob
    |
    +-- connection 1 -> segment 0
    +-- connection 2 -> segment 1
    +-- connection 3 -> segment 2
    +-- connection 4 -> segment 3
    +-- connection 5 -> segment 4
    +-- connection 6 -> segment 5
    +-- connection 7 -> segment 6
    +-- connection 8 -> segment 7
```

Each segment is independently resumable.

Inside each segment the downloader still uses small bounded Range requests, so a dead connection does not lose the entire segment.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Windows:

```powershell
python -m pip install -r requirements.txt
```

Optional aria2 backend:

```text
Install aria2c separately and make sure it is available on PATH.
```

## Download

```bash
python3 main.py qwen3.8:27b
```

Default network settings:

```text
connections:      8
range request:    8 MiB
read idle timeout: 20 seconds
retries:          10 per failed range
```

## Use more connections

Similar to an aria2 configuration such as `-x 8 -s 8`:

```bash
python3 main.py qwen3.8:27b --connections 8
```

For a very fast connection:

```bash
python3 main.py qwen3.8:27b \
  --connections 16 \
  --range-size-mib 16
```

The maximum allowed `--connections` value is 32.

More connections are not always faster. CDN throttling, server rate limiting, Wi-Fi, disk speed, and your internet link can become the bottleneck.

## Recommended unstable-connection settings

```bash
python3 main.py qwen3.8:27b \
  --connections 8 \
  --range-size-mib 8 \
  --read-timeout 20 \
  --retries 20
```

## Resume

Each parallel segment is stored separately:

```text
sha256-abc....segment-000.part
sha256-abc....segment-001.part
sha256-abc....segment-002.part
...
```

If the program stops, run the same command again:

```bash
python3 main.py qwen3.8:27b
```

Every worker resumes its own segment from its current file size.

When aria2c is used, its sequential partial file and control file are stored
as:

```text
sha256-abc....part
sha256-abc....part.aria2
```

Existing HTTPX segment state takes precedence in `auto` mode. Use
`--backend httpx` to continue an existing set of `.segment-*.part` files, or
`--backend aria2` to require aria2c for a fresh/sequential partial download.

## Upgrade from v3

v3 used a single sequential file:

```text
sha256-abc....part
```

v4 detects that file and migrates its already-downloaded bytes into the new per-segment layout, so you do not have to discard your previous partial download.

## Verification

After all segments finish:

1. Segment sizes are checked.
2. Segments are merged in the correct byte order.
3. SHA-256 is calculated while merging.
4. The expected Ollama digest is verified.
5. Only then is the final blob installed.
6. Segment files are deleted only after verification succeeds.

## Existing completed blobs

Completed blobs already in:

```text
~/.ollama/models/blobs/
```

are checked and skipped when valid.

## Model directory

The downloader uses `$OLLAMA_MODELS` when set.

Otherwise:

```text
~/.ollama/models
```

Custom directory:

```bash
python3 main.py qwen3.8:27b \
  --models-dir /path/to/ollama/models
```

Windows:

```powershell
python main.py qwen3.8:27b --models-dir "D:\ollama\models"
```

## Options

```bash
python3 main.py --help
```

Important options:

```text
--backend auto|httpx|aria2
--connections N
--range-size-mib N
--retries N
--read-timeout SECONDS
--connect-timeout SECONDS
--models-dir PATH
--registry URL
```

## Dependencies

```text
httpx
tenacity
filelock
rich
aria2c (optional executable)
```

## After download

```bash
ollama list
ollama run qwen3.8:27b
```
