import { createServer } from 'node:http';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { loadModel, completion, unloadModel } from '@qvac/sdk';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? 3002;

// Warm up bare.exe so Windows Defender/AV scans the executable at startup
// instead of on the first request (which has a 30-second RPC timeout).
warmUpBare();

const HTML = readFileSync(join(__dirname, 'public', 'index.html'));

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
      if (modelId) {
        try { await unloadModel({ modelId }); } catch { /* ignore */ }
      }
      res.end();
    }
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
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
  if (!existsSync(barePath)) return;
  const child = spawn(barePath, ['--version'], { stdio: 'ignore' });
  child.once('error', () => {}); // prevent unhandled error events
}
