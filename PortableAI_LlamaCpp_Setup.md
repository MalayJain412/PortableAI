# Portable Local LLM Setup on USB (Windows)

This document records the steps completed so far to prepare a local LLM on a USB pendrive using a **llamafile executable built around llama.cpp**.

> **Status:** The model file was downloaded successfully and its Windows executable signature was checked. The transcript does not yet confirm a successful model launch or inference, so those are listed as pending verification.

## 1. Environment

- Operating system: Windows
- USB drive: `F:`
- Project directory: `F:\PortableAI`
- USB filesystem: reformatted to **NTFS**
- Model family: Qwen3.5
- Model size class: 0.8B parameters
- Quantization/file variant: `Qwen3.5-0.8B-Q8_0`
- Runtime packaging: Mozilla AI `llamafile` (llama.cpp-based)

## 2. Create the model directory

Open PowerShell and run:

```powershell
New-Item -ItemType Directory -Force -Path `
    "F:\PortableAI\models", `
    "F:\PortableAI\agent", `
    "F:\PortableAI\logs"
Set-Location "F:\PortableAI\models"
```

The `agent` and `logs` folders were created for the planned Python application. The model download was performed from `F:\PortableAI\models`.

## 3. Download the llamafile

The model was downloaded from the Mozilla AI Hugging Face repository:

```powershell
curl.exe -L --fail --retry 3 `
  -o "Qwen3.5-0.8B-Q8_0.llamafile" `
  "https://huggingface.co/mozilla-ai/llamafile_0.10/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile"
```

The first transfer was interrupted with:

```text
curl: (56) schannel: server closed abruptly (missing close_notify)
```

At that point, the partial file was approximately **126.68 MB**. The file was not deleted; the transfer was resumed.

### Resume an interrupted download

```powershell
curl.exe -L --fail --retry 10 --retry-delay 5 --continue-at - `
  -o "Qwen3.5-0.8B-Q8_0.llamafile" `
  "https://huggingface.co/mozilla-ai/llamafile_0.10/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile"
```

The resumed download completed successfully. Curl reported approximately `1.02G` transferred for the remaining portion.

## 4. Verify the downloaded file size

Run:

```powershell
Get-Item .\Qwen3.5-0.8B-Q8_0.llamafile |
    Select-Object Name,
        Length,
        @{N="SizeGB";E={[math]::Round($_.Length / 1GB,2)}}
```

Observed result:

```text
Name                        Length      SizeGB
----                        ------      ------
Qwen3.5-0.8B-Q8_0.llamafile  1229603934  1.15
```

This confirms the local file size, but file size alone is not a cryptographic integrity check. No SHA-256 checksum comparison was recorded.

## 5. Check the executable signature

The PowerShell version on this machine did not support the `-Count` parameter for `Format-Hex`, so the first bytes were read using `Get-Content`:

```powershell
$bytes = Get-Content .\Qwen3.5-0.8B-Q8_0.llamafile `
    -Encoding Byte -TotalCount 16

$bytes | ForEach-Object { "{0:X2}" -f $_ }
```

Observed bytes:

```text
4D
5A
71
46
70
44
3D
27
0A
0A
00
10
00
F8
00
00
```

The leading bytes `4D 5A` correspond to `MZ`, the Windows executable signature.

PowerShell also recognized the file as an application when checked with:

```powershell
Get-Command .\Qwen3.5-0.8B-Q8_0.llamafile `
    -ErrorAction SilentlyContinue
```

This is a basic format check, not proof that inference works.

## 6. Rename the executable

The next instructed step was to rename the file to `.exe`:

```powershell
Rename-Item `
  "F:\PortableAI\models\Qwen3.5-0.8B-Q8_0.llamafile" `
  "Qwen3.5-0.8B-Q8_0.exe"
```

Then verify:

```powershell
Get-Item .\Qwen3.5-0.8B-Q8_0.exe |
    Select-Object Name,
        @{N="SizeGB";E={[math]::Round($_.Length / 1GB,2)}}
```

**Note:** The conversation moved on to planning the Python agent after the rename/run instructions. The transcript does not contain the command output confirming that the rename succeeded or that the executable launched. Check the actual filename before proceeding.

## 7. Planned first launch (not yet confirmed)

From PowerShell:

```powershell
Set-Location "F:\PortableAI\models"
.\Qwen3.5-0.8B-Q8_0.exe
```

The expected behavior and API must be verified against this particular build. Do not assume the server port, `/v1/models` endpoint, streaming support, or GPU acceleration until tested.

If the process starts successfully, test the actual documented/local API and record:
- Startup time
- Time to first token (TTFT)
- Generation speed (tokens/second)
- Total response latency
- CPU and GPU utilization, if available

## 8. Current project layout

The intended structure is:

```text
F:\PortableAI\
├── models\
│   └── Qwen3.5-0.8B-Q8_0.exe   # expected name after rename; verify
├── tools\
├── runtime\
├── cache\
├── agent\
└── logs\
```

The model file is confirmed at `F:\PortableAI\models\Qwen3.5-0.8B-Q8_0.llamafile` before the rename step. The `tools`, `runtime`, and `cache` directories were subsequently planned for the portable Python/uv environment; their contents have not been confirmed in the available transcript.

## 9. Important distinction: llama.cpp vs. llamafile

The completed download is a **Mozilla AI llamafile**. Llamafile packages a model together with an executable inference runtime based on llama.cpp. This is not the same workflow as separately compiling or installing the standalone `llama.cpp` project.

For this project, the intended approach is to use the downloaded llamafile executable unless a later decision explicitly switches to standalone llama.cpp.

## 10. Next steps

1. Check whether the file has been renamed to `.exe`.
2. Launch it and verify that the model loads without an error.
3. Verify a real inference request.
4. Identify the actual local API endpoint and confirm the chat-completion format.
5. Benchmark latency on the current laptop.
6. Build the Python MCP agent and Windows tools after inference is verified.
7. Test portability on a second Windows machine before calling the setup fully portable.

## References

- Mozilla AI llamafile repository: https://github.com/mozilla-ai/llamafile
- Mozilla AI llamafile quickstart: https://github.com/mozilla-ai/llamafile/blob/main/docs/quickstart.md
- Download source used: https://huggingface.co/mozilla-ai/llamafile_0.10/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile
