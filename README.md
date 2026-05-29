# QVAC Local AI

A minimal Node.js web app for local LLM inference powered by [@qvac/sdk](https://www.npmjs.com/package/@qvac/sdk).  
Inference runs entirely on your machine — no data is sent to the cloud.

## Requirements

- **Node.js** v18 or later
- **Windows x64** (the underlying `bare-runtime` and `@qvac/embed-llamacpp` native packages currently target Windows x64)
- **Git for Windows** — the postinstall script copies `libcrypto-3-x64.dll` and `libssl-3-x64.dll` from `C:\Program Files\Git\mingw64\bin` into the native addon's prebuilds directory

## Install

```
npm install
```

The `postinstall` script runs automatically and copies the required OpenSSL DLLs.

## Run

```
npm run dev
```

Open http://localhost:3002 in your browser.

To use a different port:

```
PORT=8080 npm run dev
```

## Usage

1. Pick a model from the dropdown (or paste a custom `.gguf` Hugging Face URL).
2. Type a prompt and press **▶ Ask** or **Ctrl + Enter**.
3. On the first run the model file is downloaded and cached; subsequent runs use the local cache.

## Preset models

| Model | Size |
|---|---|
| Llama 3.2 1B — Meta | ~773 MB |
| Llama 3.2 3B — Meta | ~1.9 GB |
| Qwen3 0.6B — Alibaba | ~400 MB |
| Gemma 3 1B — Google | ~670 MB |
| Mistral 7B — Mistral AI | ~4.1 GB |

## Project structure

```
server.mjs          # HTTP server — serves the UI and /api/completion
public/
  index.html        # Single-page vanilla JS UI
scripts/
  fix-dlls.cjs      # postinstall — copies OpenSSL DLLs for the native addon
```

## How it works

`server.mjs` starts a plain Node.js HTTP server. On `POST /api/completion` it calls `@qvac/sdk` which spawns a `bare.exe` worker process, loads the requested GGUF model via `llama.cpp`, and streams tokens back over a Windows named pipe. The tokens are forwarded to the browser as a chunked HTTP response and rendered incrementally.
