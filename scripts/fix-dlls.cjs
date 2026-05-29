// Copies OpenSSL DLLs that @qvac/embed-llamacpp requires on Windows.
// Git for Windows ships them at mingw64/bin — they're not in the system PATH by default.
const { existsSync, copyFileSync } = require('fs');
const { join } = require('path');

const GIT_DIR = 'C:\\Program Files\\Git\\mingw64\\bin';
const TARGET_DIR = join(__dirname, '..', 'node_modules', '@qvac', 'embed-llamacpp', 'prebuilds', 'win32-x64');
const DLLS = ['libcrypto-3-x64.dll', 'libssl-3-x64.dll'];

if (!existsSync(TARGET_DIR)) {
  console.log('ℹ️  @qvac/embed-llamacpp not found, skipping DLL fix.');
  process.exit(0);
}

for (const dll of DLLS) {
  const src = join(GIT_DIR, dll);
  const dst = join(TARGET_DIR, dll);
  if (!existsSync(src)) {
    console.warn(`⚠️  ${src} not found — install Git for Windows or copy ${dll} to:\n   ${TARGET_DIR}`);
    continue;
  }
  try {
    copyFileSync(src, dst);
    console.log(`✅ Copied ${dll}`);
  } catch (e) {
    console.warn(`⚠️  Could not copy ${dll}: ${e.message}`);
  }
}
