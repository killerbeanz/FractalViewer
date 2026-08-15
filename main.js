import { COMPUTE_WGSL, RENDER_WGSL } from './shaders.js';

// PixelState struct layout (must match WGSL exactly, std430-ish storage rules):
// a: vec2<f32>          -> 8 bytes,  offset 0   (Brent's "hare")
// lnBase: vec2<f32>     -> 8 bytes,  offset 8
// tortoise: vec2<f32>   -> 8 bytes,  offset 16
// power: u32            -> 4 bytes,  offset 24
// lam: u32              -> 4 bytes,  offset 28
// totalIter: u32        -> 4 bytes,  offset 32
// status: u32           -> 4 bytes,  offset 36
// initialized: u32      -> 4 bytes,  offset 40 -> total 44, padded to 8-byte multiple -> 48
const PIXEL_STATE_BYTES = 48;

const canvas = document.getElementById('gpu-canvas');
const statusEl = document.getElementById('status-text');
const progressBar = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const centerXInput = document.getElementById('center-x');
const centerYInput = document.getElementById('center-y');
const scaleInput = document.getElementById('scale');
const maxIterInput = document.getElementById('max-iter');
const resolutionSelect = document.getElementById('resolution');
const recomputeBtn = document.getElementById('recompute-btn');
const resetViewBtn = document.getElementById('reset-view-btn');

const DEFAULT_VIEW = { centerX: 0.0, centerY: 0.0, scale: 2.5 };
let view = { ...DEFAULT_VIEW };

let device, context, presentFormat;
let computePipeline, renderPipeline;
let stateBuffer, paramsBuffer, doneCounterBuffer, doneReadbackBuffer;
let outputTexture, sampler;
let computeBindGroup, renderBindGroup;
let gridWidth = 0, gridHeight = 0;
let totalPixels = 0;
let running = false;
let mapInFlight = false;
let animationHandle = null;

async function initGPU() {
  if (!navigator.gpu) {
    statusEl.textContent = 'WebGPU is not available in this browser. Try a recent Chrome, Edge, or Firefox.';
    throw new Error('WebGPU unavailable');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    statusEl.textContent = 'No compatible GPU adapter found.';
    throw new Error('No adapter');
  }
  // The default device limits are conservative (128MB max storage buffer
  // binding), but our per-pixel state buffer easily exceeds that at full
  // resolution. Request the adapter's actual supported limits instead.
  device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) => {
    statusEl.textContent = `GPU device lost: ${info.message}. Reload to restart.`;
  });

  context = canvas.getContext('webgpu');
  presentFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: presentFormat, alphaMode: 'opaque' });

  const computeModule = device.createShaderModule({ code: COMPUTE_WGSL });
  computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'main' },
  });

  const renderModule = device.createShaderModule({ code: RENDER_WGSL });
  renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: presentFormat }] },
    primitive: { topology: 'triangle-list' },
  });

  sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });
}

function resolutionScale() {
  return parseFloat(resolutionSelect.value);
}

function resizeCanvasAndBuffers() {
  const rect = canvas.getBoundingClientRect();
  const scale = resolutionScale();
  gridWidth = Math.max(1, Math.floor(rect.width * scale));
  gridHeight = Math.max(1, Math.floor(rect.height * scale));
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  totalPixels = gridWidth * gridHeight;

  // Safety net: if this grid's state buffer would exceed what this GPU
  // actually supports (varies by hardware), scale the grid down to fit
  // rather than crashing.
  const maxBindingBytes = device.limits.maxStorageBufferBindingSize;
  const maxPixels = Math.floor(maxBindingBytes / PIXEL_STATE_BYTES);
  if (totalPixels > maxPixels) {
    const shrink = Math.sqrt(maxPixels / totalPixels);
    gridWidth = Math.max(1, Math.floor(gridWidth * shrink));
    gridHeight = Math.max(1, Math.floor(gridHeight * shrink));
    totalPixels = gridWidth * gridHeight;
    statusEl.textContent = 'Resolution capped — this GPU has a smaller max buffer size.';
  }

  if (stateBuffer) stateBuffer.destroy();
  if (outputTexture) outputTexture.destroy();

  stateBuffer = device.createBuffer({
    size: PIXEL_STATE_BYTES * totalPixels,
    usage: GPUBufferUsage.STORAGE,
  });

  outputTexture = device.createTexture({
    size: { width: gridWidth, height: gridHeight },
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST,
  });

  if (!paramsBuffer) {
    paramsBuffer = device.createBuffer({
      size: 32, // matches Params struct, padded to 16-byte multiple
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }
  if (!doneCounterBuffer) {
    doneCounterBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }
  if (!doneReadbackBuffer) {
    doneReadbackBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: stateBuffer } },
      { binding: 1, resource: outputTexture.createView() },
      { binding: 2, resource: { buffer: paramsBuffer } },
      { binding: 3, resource: { buffer: doneCounterBuffer } },
    ],
  });

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: outputTexture.createView() },
      { binding: 1, resource: sampler },
    ],
  });
}

function writeParams() {
  const maxIter = parseInt(maxIterInput.value, 10);
  const itersPerPass = 10;

  const buf = new ArrayBuffer(32);
  const dv = new DataView(buf);
  dv.setUint32(0, gridWidth, true);
  dv.setUint32(4, gridHeight, true);
  dv.setUint32(8, maxIter, true);
  dv.setUint32(12, itersPerPass, true);
  dv.setFloat32(16, view.centerX, true);
  dv.setFloat32(20, view.centerY, true);
  dv.setFloat32(24, view.scale, true);
  // offset 28 is an unused pad float to keep the struct a clean 32 bytes
  device.queue.writeBuffer(paramsBuffer, 0, buf);
}

function resetAndStart() {
  centerXInput.value = view.centerX;
  centerYInput.value = view.centerY;
  scaleInput.value = view.scale;

  resizeCanvasAndBuffers();
  writeParams();
  device.queue.writeBuffer(doneCounterBuffer, 0, new Uint32Array([0]));

  running = true;
  mapInFlight = false;
  updateProgressUI(0);
  statusEl.textContent = 'Computing…';
  if (!animationHandle) animationHandle = requestAnimationFrame(frame);
}

function updateProgressUI(done) {
  const pct = totalPixels > 0 ? Math.min(100, (done / totalPixels) * 100) : 0;
  progressBar.style.width = `${pct.toFixed(1)}%`;
  progressLabel.textContent = `${pct.toFixed(1)}%  (${done.toLocaleString()} / ${totalPixels.toLocaleString()} px)`;
}

function maybeReadProgress() {
  if (mapInFlight) return;
  mapInFlight = true;
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(doneCounterBuffer, 0, doneReadbackBuffer, 0, 4);
  device.queue.submit([encoder.finish()]);

  doneReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
    const done = new Uint32Array(doneReadbackBuffer.getMappedRange())[0];
    doneReadbackBuffer.unmap();
    mapInFlight = false;
    updateProgressUI(done);
    if (done >= totalPixels) {
      running = false;
      statusEl.textContent = 'Done';
    }
  }).catch(() => { mapInFlight = false; });
}

let frameCount = 0;

function frame() {
  if (running) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, computeBindGroup);
    pass.dispatchWorkgroups(Math.ceil(gridWidth / 8), Math.ceil(gridHeight / 8));
    pass.end();
    device.queue.submit([encoder.finish()]);

    frameCount++;
    if (frameCount % 8 === 0) maybeReadProgress();
  }

  // always blit the latest output texture to the canvas
  const encoder2 = device.createCommandEncoder();
  const view2 = context.getCurrentTexture().createView();
  const renderPass = encoder2.beginRenderPass({
    colorAttachments: [{ view: view2, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
  });
  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBindGroup);
  renderPass.draw(3);
  renderPass.end();
  device.queue.submit([encoder2.finish()]);

  animationHandle = requestAnimationFrame(frame);
}

// --- interaction: drag to pan, wheel to zoom toward cursor ---
// Note: recomputing means recreating the per-pixel state buffer and
// restarting the whole multi-pass iteration, so we deliberately don't
// trigger a real recompute on every pointermove — that would recreate GPU
// buffers 100+ times/sec while dragging. Instead we give a cheap CSS-only
// pan preview during the drag and only recompute once, on release.
let dragging = false;
let dragStartX = 0, dragStartY = 0;
let pendingCenterX = 0, pendingCenterY = 0;

canvas.addEventListener('pointerdown', (e) => {
  dragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  pendingCenterX = view.centerX;
  pendingCenterY = view.centerY;
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const rect = canvas.getBoundingClientRect();
  const pixelDx = e.clientX - dragStartX;
  const pixelDy = e.clientY - dragStartY;
  canvas.style.transform = `translate(${pixelDx}px, ${pixelDy}px)`;

  const dx = pixelDx / rect.height * 2.0 * view.scale;
  const dy = pixelDy / rect.height * 2.0 * view.scale;
  pendingCenterX = view.centerX - dx;
  pendingCenterY = view.centerY + dy;
});

canvas.addEventListener('pointerup', (e) => {
  if (!dragging) return;
  dragging = false;
  canvas.releasePointerCapture(e.pointerId);
  canvas.style.transform = 'none';
  view.centerX = pendingCenterX;
  view.centerY = pendingCenterY;
  resetAndStart();
});

let wheelDebounce = null;
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const aspect = rect.width / rect.height;
  const u = ((e.clientX - rect.left) / rect.width - 0.5) * 2.0 * aspect;
  const v = ((e.clientY - rect.top) / rect.height - 0.5) * 2.0;
  const zBeforeX = view.centerX + u * view.scale;
  const zBeforeY = view.centerY - v * view.scale;

  const zoomFactor = e.deltaY < 0 ? 0.85 : 1 / 0.85;
  view.scale *= zoomFactor;

  view.centerX = zBeforeX - u * view.scale;
  view.centerY = zBeforeY + v * view.scale;

  // coalesce rapid trackpad wheel events into a single recompute
  clearTimeout(wheelDebounce);
  wheelDebounce = setTimeout(resetAndStart, 120);
}, { passive: false });

recomputeBtn.addEventListener('click', () => {
  view.centerX = parseFloat(centerXInput.value);
  view.centerY = parseFloat(centerYInput.value);
  view.scale = parseFloat(scaleInput.value);
  resetAndStart();
});

resetViewBtn.addEventListener('click', () => {
  view = { ...DEFAULT_VIEW };
  resetAndStart();
});

resolutionSelect.addEventListener('change', resetAndStart);

window.addEventListener('resize', () => {
  if (!device) return; // ignore resizes that fire before GPU init finishes
  resetAndStart();
});

(async function boot() {
  try {
    await initGPU();
    resetAndStart();
  } catch (err) {
    console.error(err);
  }
})();
