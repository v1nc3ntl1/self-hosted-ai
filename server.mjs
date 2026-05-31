import { createServer } from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, spawnSync } from 'node:child_process';
import {
  loadModel, completion, unloadModel, diffusion,
  ragIngest, ragSearch,
  close as closeBareWorker,
  SD_V2_1_1B_Q4_0,
  FLUX_2_KLEIN_4B_Q4_0, FLUX_2_KLEIN_4B_VAE, QWEN3_4B_Q4_K_M,
} from '@qvac/sdk';

// Direct HTTPS downloads — avoids the registry:// P2P protocol that requires Hypercore peers
const EMBEDDING_MODEL_URL = 'https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300m-Q4_0.gguf';
const SD21_MODEL_URL = 'https://huggingface.co/gpustack/stable-diffusion-v2-1-GGUF/resolve/12ddc22724f6da35f0b6006e459fae66eaf56931/stable-diffusion-v2-1-Q4_0.gguf';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? 3003;

// Native addons are warmed up below (before server.listen) so the port
// only opens once Windows Defender has scanned all .bare addon files.

const HTML = readFileSync(join(__dirname, 'public', 'index.html'));

// unloadModel can hang on 'Closing IPC server' — cap it so res.end() is always called.
// After every unload, re-run the Bare warmup in the background so Defender pre-scans
// the native addons before the next request spawns a fresh Bare worker.
const unloadWithTimeout = (modelId, ms = 5000) =>
  Promise.race([
    unloadModel({ modelId }).catch(() => {}),
    new Promise(r => setTimeout(r, ms)),
  ]).finally(() => warmUpBare().catch(() => {}));

const server = createServer(async (req, res) => {
  // ── Serve the UI ──────────────────────────────────────────────────────────
  if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html')) {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(HTML);
    return;
  }

  // ── Completion API ────────────────────────────────────────────────────────
  if (req.method === 'POST' && req.url === '/api/completion') {
    let body = '';
    for await (const chunk of req) body += chunk;

    let prompt, modelSrc;
    try {
      ({ prompt, modelSrc } = JSON.parse(body));
    } catch {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Invalid JSON body');
      return;
    }

    if (!prompt?.trim() || !modelSrc?.trim()) {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Missing prompt or modelSrc');
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'text/plain; charset=utf-8',
      'Transfer-Encoding': 'chunked',
      'Cache-Control': 'no-cache',
    });

    let modelId;
    try {
      modelId = await loadModel({ modelSrc, modelType: 'llamacpp-completion' });
      const result = completion({
        modelId,
        history: [{ role: 'user', content: prompt }],
        stream: true,
      });
      for await (const token of result.tokenStream) {
        res.write(token);
      }
    } catch (err) {
      res.write(`\n\nError: ${err.message}`);
    } finally {
      if (modelId) await unloadWithTimeout(modelId);
      res.end();
    }
    return;
  }

  // ── Image generation API (SSE) ──────────────────────────────────────────
  if (req.method === 'POST' && req.url === '/api/image') {
    let body = '';
    for await (const chunk of req) body += chunk;

    let prompt, modelPreset, width, height, steps;
    try {
      ({ prompt, modelPreset, width, height, steps } = JSON.parse(body));
    } catch {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Invalid JSON body');
      return;
    }

    if (!prompt?.trim()) {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Missing prompt');
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    const sse = (event, data) =>
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

    const isFlux = modelPreset === 'flux2klein';
    let modelId;
    try {
      modelId = await loadModel({
        modelSrc: isFlux ? FLUX_2_KLEIN_4B_Q4_0 : SD21_MODEL_URL,
        modelType: isFlux ? 'diffusion' : 'sdcpp-generation',
        modelConfig: isFlux
          ? { device: 'gpu', threads: 4, llmModelSrc: QWEN3_4B_Q4_K_M, vaeModelSrc: FLUX_2_KLEIN_4B_VAE }
          : { prediction: 'v', device: 'gpu' },
        onProgress: (p) => sse('load', { percentage: +p.percentage.toFixed(1) }),
      });

      const result = diffusion({
        modelId,
        prompt,
        width:  width  || 512,
        height: height || 512,
        steps:  steps  || 20,
        guidance:  isFlux ? 3.5 : 7.5,
        cfg_scale: isFlux ? 1   : 7.5,
        seed: -1,
      });

      for await (const { step, totalSteps } of result.progressStream) {
        sse('progress', { step, totalSteps });
      }

      const buffers = await result.outputs;
      sse('image', { png: Buffer.from(buffers[0]).toString('base64') });
    } catch (err) {
      sse('error', { message: err.message });
    } finally {
      if (modelId) await unloadWithTimeout(modelId);
      sse('done', {});
      res.end();
    }
    return;
  }

  // ── RAG ingest ────────────────────────────────────────────────────────
  if (req.method === 'POST' && req.url === '/api/rag/ingest') {
    let body = '';
    for await (const chunk of req) body += chunk;

    let documents, workspace;
    try {
      ({ documents, workspace = 'default' } = JSON.parse(body));
    } catch {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Invalid JSON body');
      return;
    }

    if (!Array.isArray(documents) || documents.length === 0) {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Missing or empty documents array');
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    const sseI = (event, data) =>
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

    let embModelId;
    try {
      embModelId = await loadModel({
        modelSrc: EMBEDDING_MODEL_URL,
        modelType: 'llamacpp-embedding',
        onProgress: (p) => sseI('load', { percentage: +p.percentage.toFixed(1) }),
      });
      await ragIngest({
        modelId: embModelId,
        documents,
        workspace,
        onProgress: (stage, current, total) => sseI('progress', { stage, current, total }),
      });
      sseI('done', { processed: documents.length });
    } catch (err) {
      sseI('error', { message: err.message });
    } finally {
      if (embModelId) await unloadWithTimeout(embModelId);
      res.end();
    }
    return;
  }

  // ── RAG query ─────────────────────────────────────────────────────────
  if (req.method === 'POST' && req.url === '/api/rag/query') {
    let body = '';
    for await (const chunk of req) body += chunk;

    let query, completionModelSrc, topK, workspace;
    try {
      ({ query, completionModelSrc, topK = 3, workspace = 'default' } = JSON.parse(body));
    } catch {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Invalid JSON body');
      return;
    }

    if (!query?.trim() || !completionModelSrc?.trim()) {
      res.writeHead(400, { 'Content-Type': 'text/plain' }).end('Missing query or completionModelSrc');
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    const sseQ = (event, data) =>
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

    let embModelId, compModelId;
    try {
      sseQ('status', { message: 'Loading embedding model…' });
      embModelId = await loadModel({
        modelSrc: EMBEDDING_MODEL_URL,
        modelType: 'llamacpp-embedding',
        onProgress: (p) => sseQ('load', { percentage: +p.percentage.toFixed(1), phase: 'embed' }),
      });

      sseQ('status', { message: 'Searching knowledge base…' });
      const results = await ragSearch({ modelId: embModelId, query, topK, workspace });
      await unloadWithTimeout(embModelId);
      embModelId = null;

      sseQ('context', { results: results.map(r => ({ content: r.content, score: r.score })) });

      if (results.length === 0) {
        sseQ('status', { message: 'No documents found — ingest some first.' });
        sseQ('done', {});
        res.end();
        return;
      }

      const context = results
        .map((r, i) => `[${i + 1}] ${r.content}`)
        .join('\n\n');
      const augmentedPrompt =
        `Answer the following question using only the provided context. ` +
        `If the context doesn't contain enough information, say so.\n\n` +
        `Context:\n${context}\n\nQuestion: ${query}`;

      sseQ('status', { message: 'Loading completion model…' });
      compModelId = await loadModel({
        modelSrc: completionModelSrc,
        modelType: 'llamacpp-completion',
        onProgress: (p) => sseQ('load', { percentage: +p.percentage.toFixed(1), phase: 'completion' }),
      });

      sseQ('status', { message: 'Generating answer…' });
      const result = completion({
        modelId: compModelId,
        history: [{ role: 'user', content: augmentedPrompt }],
        stream: true,
      });
      for await (const token of result.tokenStream) {
        sseQ('token', { text: token });
      }
      sseQ('done', {});
    } catch (err) {
      sseQ('error', { message: err.message });
    } finally {
      if (embModelId) await unloadWithTimeout(embModelId);
      if (compModelId) await unloadWithTimeout(compModelId);
      res.end();
    }
    return;
  }

  // ── Reset (kill Bare worker to free GPU/memory) ────────────────────────
  if (req.method === 'POST' && req.url === '/api/reset') {
    try {
      await Promise.race([
        closeBareWorker(),
        new Promise(r => setTimeout(r, 6000)),
      ]);
      warmUpBare().catch(() => {});
      res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ ok: true }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' }).end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
});

console.log('Warming up native addons…');
await warmUpBare().catch(() => {});

server.on('error', (err) => {
  if (err.code !== 'EADDRINUSE') throw err;
  console.log(`Port ${PORT} is in use — killing existing process…`);
  try {
    if (process.platform === 'win32') {
      spawnSync('powershell', [
        '-Command',
        `Get-NetTCPConnection -LocalPort ${PORT} -State Listen | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }`,
      ], { stdio: 'ignore' });
    } else {
      spawnSync('sh', ['-c', `lsof -ti:${PORT} | xargs kill -9`], { stdio: 'ignore' });
    }
  } catch { /* already gone */ }
  setTimeout(() => server.listen(PORT), 500);
});

server.listen(PORT, () => {
  console.log(`🤖 QVAC Local AI  →  http://localhost:${PORT}`);
});

function warmUpBare() {
  const ext = process.platform === 'win32' ? '.exe' : '';
  const barePath = join(
    __dirname,
    'node_modules',
    `bare-runtime-${process.platform}-${process.arch}`,
    'bin',
    `bare${ext}`,
  );
  if (!existsSync(barePath)) return Promise.resolve();

  const spawnAndWait = (args) => new Promise((resolve) => {
    const child = spawn(barePath, args, { stdio: 'ignore', cwd: __dirname });
    child.once('error', resolve);
    child.once('exit', resolve);
  });

  // Run bare.exe and all native addons concurrently so Defender scans them in
  // parallel. Each process exits as soon as the addon is loaded.
  const bindings = [
    join(__dirname, 'node_modules', '@qvac', 'llm-llamacpp',             'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'embed-llamacpp',           'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'diffusion-cpp',            'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'transcription-whispercpp', 'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'transcription-parakeet',   'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'translation-nmtcpp',       'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'tts-onnx',                 'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'ocr-onnx',                 'binding.js'),
    join(__dirname, 'node_modules', '@qvac', 'onnx',                     'binding.js'),
  ].filter(existsSync);

  return Promise.all([
    spawnAndWait(['--version']),
    ...bindings.map(b => spawnAndWait([b])),
  ]);
}
