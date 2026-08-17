import { COMPUTE_WGSL, RENDER_WGSL } from './shaders.js';

const PIXEL_STATE_BYTES = 48;
const MAX_GRADIENT_STOPS = 16;
const GRADIENT_BUFFER_BYTES = 336;

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

const gradientTab = document.getElementById('gradient-tab');
const gradientEditor = document.getElementById('gradient-editor');
const gradientClose = document.getElementById('gradient-close');
const gradientPreview = document.getElementById('gradient-preview');
const gradientAlgorithm = document.getElementById('gradient-algorithm');
const gradientLengthInput = document.getElementById('gradient-length');
const gradientStopsEl = document.getElementById('gradient-stops');
const gradientAddStop = document.getElementById('gradient-add-stop');

const DEFAULT_VIEW = {
  centerX: 0.0,
  centerY: 0.0,
  scale: 2.5
};

let view = { ...DEFAULT_VIEW };

let gradient = {
  algorithm: 'linear',
  length: 1000,
  stops: [
    { color: '#18003d', weight: 1 },
    { color: '#7200d4', weight: 1 },
    { color: '#0066ff', weight: 1 },
    { color: '#00d9ff', weight: 1 },
    { color: '#fff4c2', weight: 1 }
  ]
};

let device;
let context;
let presentFormat;
let computePipeline;
let renderPipeline;
let stateBuffer;
let paramsBuffer;
let doneCounterBuffer;
let doneReadbackBuffer;
let gradientBuffer;
let outputTexture;
let sampler;
let computeBindGroup;
let renderBindGroup;
let gridWidth = 0;
let gridHeight = 0;
let totalPixels = 0;
let running = false;
let mapInFlight = false;
let animationHandle = null;
let frameCount = 0;
let gradientUpdateTimer = null;

async function initGPU() {
  if (!navigator.gpu) {
    statusEl.textContent =
      'WebGPU is not available in this browser. Try a recent Chrome, Edge, or Firefox.';
    throw new Error('WebGPU unavailable');
  }

  const adapter = await navigator.gpu.requestAdapter();

  if (!adapter) {
    statusEl.textContent = 'No compatible GPU adapter found.';
    throw new Error('No adapter');
  }

  device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize
    }
  });

  device.lost.then((info) => {
    statusEl.textContent = `GPU device lost: ${info.message}. Reload to restart.`;
  });

  context = canvas.getContext('webgpu');
  presentFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentFormat,
    alphaMode: 'opaque'
  });

  const computeModule = device.createShaderModule({
    code: COMPUTE_WGSL
  });

  computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: computeModule,
      entryPoint: 'main'
    }
  });

  const renderModule = device.createShaderModule({
    code: RENDER_WGSL
  });

  renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vs_main'
    },
    fragment: {
      module: renderModule,
      entryPoint: 'fs_main',
      targets: [{ format: presentFormat }]
    },
    primitive: {
      topology: 'triangle-list'
    }
  });

  sampler = device.createSampler({
    magFilter: 'nearest',
    minFilter: 'nearest'
  });
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

  const maxBindingBytes = device.limits.maxStorageBufferBindingSize;
  const maxPixels = Math.floor(maxBindingBytes / PIXEL_STATE_BYTES);

  if (totalPixels > maxPixels) {
    const shrink = Math.sqrt(maxPixels / totalPixels);

    gridWidth = Math.max(1, Math.floor(gridWidth * shrink));
    gridHeight = Math.max(1, Math.floor(gridHeight * shrink));
    totalPixels = gridWidth * gridHeight;

    statusEl.textContent =
      'Resolution capped — this GPU has a smaller max buffer size.';
  }

  if (stateBuffer) {
    stateBuffer.destroy();
  }

  if (outputTexture) {
    outputTexture.destroy();
  }

  stateBuffer = device.createBuffer({
    size: PIXEL_STATE_BYTES * totalPixels,
    usage: GPUBufferUsage.STORAGE
  });

  outputTexture = device.createTexture({
    size: {
      width: gridWidth,
      height: gridHeight
    },
    format: 'rgba8unorm',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST
  });

  if (!paramsBuffer) {
    paramsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
  }

  if (!doneCounterBuffer) {
    doneCounterBuffer = device.createBuffer({
      size: 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC
    });
  }

  if (!doneReadbackBuffer) {
    doneReadbackBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
  }

  if (!gradientBuffer) {
    gradientBuffer = device.createBuffer({
      size: GRADIENT_BUFFER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
  }

  computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: {
          buffer: stateBuffer
        }
      },
      {
        binding: 1,
        resource: outputTexture.createView()
      },
      {
        binding: 2,
        resource: {
          buffer: paramsBuffer
        }
      },
      {
        binding: 3,
        resource: {
          buffer: doneCounterBuffer
        }
      },
      {
        binding: 4,
        resource: {
          buffer: gradientBuffer
        }
      }
    ]
  });

  renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: outputTexture.createView()
      },
      {
        binding: 1,
        resource: sampler
      }
    ]
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

  device.queue.writeBuffer(paramsBuffer, 0, buf);
}

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);

  return [
    ((value >> 16) & 255) / 255,
    ((value >> 8) & 255) / 255,
    (value & 255) / 255
  ];
}

function algorithmId() {
  switch (gradient.algorithm) {
    case 'smoothstep':
      return 1;
    case 'smootherstep':
      return 2;
    case 'cosine':
      return 3;
    default:
      return 0;
  }
}

function writeGradientBuffer() {
  if (!device || !gradientBuffer) {
    return;
  }

  const buffer = new ArrayBuffer(GRADIENT_BUFFER_BYTES);

  const colors = new Float32Array(
    buffer,
    0,
    MAX_GRADIENT_STOPS * 4
  );

  const weights = new Float32Array(
    buffer,
    MAX_GRADIENT_STOPS * 4 * 4,
    MAX_GRADIENT_STOPS
  );

  const metadataOffset = MAX_GRADIENT_STOPS * 5 * 4;

  const metadata = new Uint32Array(
    buffer,
    metadataOffset,
    4
  );

  const lengthView = new Float32Array(
    buffer,
    metadataOffset + 8,
    1
  );

  const count = Math.min(
    gradient.stops.length,
    MAX_GRADIENT_STOPS
  );

  for (let i = 0; i < count; i++) {
    const rgb = hexToRgb(gradient.stops[i].color);

    colors[i * 4] = rgb[0];
    colors[i * 4 + 1] = rgb[1];
    colors[i * 4 + 2] = rgb[2];
    colors[i * 4 + 3] = 1;

    weights[i] = Math.max(
      0.0001,
      Number(gradient.stops[i].weight) || 0.0001
    );
  }

  metadata[0] = count;
  metadata[1] = algorithmId();
  lengthView[0] = Math.max(
    1,
    Number(gradient.length) || 1
  );

  device.queue.writeBuffer(
    gradientBuffer,
    0,
    buffer
  );
}

function resetAndStart() {
  centerXInput.value = view.centerX;
  centerYInput.value = view.centerY;
  scaleInput.value = view.scale;

  resizeCanvasAndBuffers();
  writeParams();
  writeGradientBuffer();

  device.queue.writeBuffer(
    doneCounterBuffer,
    0,
    new Uint32Array([0])
  );

  running = true;
  mapInFlight = false;
  frameCount = 0;

  updateProgressUI(0);
  statusEl.textContent = 'Computing…';

  if (!animationHandle) {
    animationHandle = requestAnimationFrame(frame);
  }
}

function updateProgressUI(done) {
  const pct =
    totalPixels > 0
      ? Math.min(100, (done / totalPixels) * 100)
      : 0;

  progressBar.style.width = `${pct.toFixed(1)}%`;

  progressLabel.textContent =
    `${pct.toFixed(1)}%  ` +
    `(${done.toLocaleString()} / ${totalPixels.toLocaleString()} px)`;
}

function maybeReadProgress() {
  if (mapInFlight) {
    return;
  }

  mapInFlight = true;

  const encoder = device.createCommandEncoder();

  encoder.copyBufferToBuffer(
    doneCounterBuffer,
    0,
    doneReadbackBuffer,
    0,
    4
  );

  device.queue.submit([encoder.finish()]);

  doneReadbackBuffer.mapAsync(GPUMapMode.READ)
    .then(() => {
      const done =
        new Uint32Array(
          doneReadbackBuffer.getMappedRange()
        )[0];

      doneReadbackBuffer.unmap();

      mapInFlight = false;
      updateProgressUI(done);

      if (done >= totalPixels) {
        running = false;
        statusEl.textContent = 'Done';
      }
    })
    .catch(() => {
      mapInFlight = false;
    });
}

function frame() {
  if (running) {
    const encoder = device.createCommandEncoder();

    const pass = encoder.beginComputePass();

    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, computeBindGroup);

    pass.dispatchWorkgroups(
      Math.ceil(gridWidth / 8),
      Math.ceil(gridHeight / 8)
    );

    pass.end();

    device.queue.submit([
      encoder.finish()
    ]);

    frameCount++;

    if (frameCount % 8 === 0) {
      maybeReadProgress();
    }
  }

  const encoder2 = device.createCommandEncoder();

  const view2 =
    context.getCurrentTexture().createView();

  const renderPass = encoder2.beginRenderPass({
    colorAttachments: [
      {
        view: view2,
        clearValue: {
          r: 0,
          g: 0,
          b: 0,
          a: 1
        },
        loadOp: 'clear',
        storeOp: 'store'
      }
    ]
  });

  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBindGroup);
  renderPass.draw(3);
  renderPass.end();

  device.queue.submit([
    encoder2.finish()
  ]);

  animationHandle = requestAnimationFrame(frame);
}

function interpolationAmount(t) {
  switch (gradient.algorithm) {
    case 'smoothstep':
      return t * t * (3 - 2 * t);

    case 'smootherstep':
      return t * t * t * (t * (t * 6 - 15) + 10);

    case 'cosine':
      return (1 - Math.cos(Math.PI * t)) * 0.5;

    default:
      return t;
  }
}

function drawGradientPreview() {
  const ctx = gradientPreview.getContext('2d');

  const width = gradientPreview.width;
  const height = gradientPreview.height;

  const cx = width * 0.5;
  const cy = height * 0.5;
  const radius = Math.min(width, height) * 0.4;

  ctx.clearRect(0, 0, width, height);

  const totalWeight = gradient.stops.reduce(
    (sum, stop) =>
      sum +
      Math.max(
        0.0001,
        Number(stop.weight) || 0.0001
      ),
    0
  );

  const boundaries = [];
  let accumulated = 0;

  for (const stop of gradient.stops) {
    boundaries.push(
      accumulated / totalWeight
    );

    accumulated += Math.max(
      0.0001,
      Number(stop.weight) || 0.0001
    );
  }

  const image = ctx.createImageData(
    width,
    height
  );

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;

      const distance =
        Math.sqrt(dx * dx + dy * dy);

      if (distance > radius) {
        continue;
      }

      let angle =
        Math.atan2(dy, dx) /
        (Math.PI * 2);

      angle = (angle + 1) % 1;

      let segment = 0;

      for (let i = 0; i < boundaries.length; i++) {
        const next =
          i === boundaries.length - 1
            ? 1
            : boundaries[i + 1];

        if (
          angle >= boundaries[i] &&
          angle < next
        ) {
          segment = i;
          break;
        }
      }

      const start = boundaries[segment];

      const end =
        segment === boundaries.length - 1
          ? 1
          : boundaries[segment + 1];

      const segmentSize =
        Math.max(0.000001, end - start);

      const localT =
        (angle - start) / segmentSize;

      const t =
        interpolationAmount(
          Math.max(0, Math.min(1, localT))
        );

      const a =
        hexToRgb(
          gradient.stops[segment].color
        );

      const nextIndex =
        (segment + 1) %
        gradient.stops.length;

      const b =
        hexToRgb(
          gradient.stops[nextIndex].color
        );

      const r =
        a[0] +
        (b[0] - a[0]) * t;

      const g =
        a[1] +
        (b[1] - a[1]) * t;

      const blue =
        a[2] +
        (b[2] - a[2]) * t;

      const index =
        (y * width + x) * 4;

      image.data[index] = r * 255;
      image.data[index + 1] = g * 255;
      image.data[index + 2] = blue * 255;
      image.data[index + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);

  drawGradientBoundaries(
    ctx,
    boundaries,
    cx,
    cy,
    radius
  );
}

function drawGradientBoundaries(
  ctx,
  boundaries,
  cx,
  cy,
  radius
) {
  ctx.save();

  ctx.lineWidth = 4;

  for (let i = 0; i < boundaries.length; i++) {
    const angle =
      boundaries[i] *
      Math.PI *
      2;

    const innerRadius =
      radius * 0.78;

    const outerRadius =
      radius * 1.03;

    const x1 =
      cx +
      Math.cos(angle) *
      innerRadius;

    const y1 =
      cy +
      Math.sin(angle) *
      innerRadius;

    const x2 =
      cx +
      Math.cos(angle) *
      outerRadius;

    const y2 =
      cy +
      Math.sin(angle) *
      outerRadius;

    ctx.strokeStyle =
      gradient.stops[i].color;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  ctx.restore();
}

function renderGradientStops() {
  gradientStopsEl.innerHTML = '';

  gradient.stops.forEach((stop, index) => {
    const row =
      document.createElement('div');

    row.className =
      'gradient-stop';

    const color =
      document.createElement('input');

    color.type = 'color';
    color.value = stop.color;

    const label =
      document.createElement('span');

    label.className =
      'gradient-stop-label';

    label.textContent =
      `color ${index + 1}`;

    const weight =
      document.createElement('input');

    weight.type = 'number';
    weight.className =
      'gradient-weight';

    weight.min = '0.01';
    weight.step = '0.1';
    weight.value = stop.weight;

    const remove =
      document.createElement('button');

    remove.className =
      'remove-stop';

    remove.textContent = '×';

    remove.disabled =
      gradient.stops.length <= 2;

    color.addEventListener(
      'input',
      () => {
        stop.color = color.value;
        updateGradient();
      }
    );

    weight.addEventListener(
      'input',
      () => {
        stop.weight =
          Math.max(
            0.01,
            Number(weight.value) || 0.01
          );

        updateGradient();
      }
    );

    remove.addEventListener(
      'click',
      () => {
        if (gradient.stops.length <= 2) {
          return;
        }

        gradient.stops.splice(
          index,
          1
        );

        renderGradientStops();
        updateGradient();
      }
    );

    row.append(
      color,
      label,
      weight,
      remove
    );

    gradientStopsEl.appendChild(row);
  });

  gradientAddStop.disabled =
    gradient.stops.length >=
    MAX_GRADIENT_STOPS;
}

function updateGradient() {
  gradient.algorithm =
    gradientAlgorithm.value;

  gradient.length =
    Math.max(
      1,
      Number(gradientLengthInput.value) || 1
    );

  writeGradientBuffer();
  drawGradientPreview();

  if (device && gradientBuffer) {
    clearTimeout(gradientUpdateTimer);

    gradientUpdateTimer =
      setTimeout(() => {
        resetAndStart();
      }, 80);
  }
}

gradientTab.addEventListener(
  'click',
  () => {
    gradientEditor.classList.toggle('open');
  }
);

gradientClose.addEventListener(
  'click',
  () => {
    gradientEditor.classList.remove('open');
  }
);

gradientAddStop.addEventListener(
  'click',
  () => {
    if (
      gradient.stops.length >=
      MAX_GRADIENT_STOPS
    ) {
      return;
    }

    gradient.stops.push({
      color: '#ffffff',
      weight: 1
    });

    renderGradientStops();
    updateGradient();
  }
);

gradientAlgorithm.addEventListener(
  'change',
  updateGradient
);

gradientLengthInput.addEventListener(
  'input',
  updateGradient
);

let dragging = false;
let dragStartX = 0;
let dragStartY = 0;
let pendingCenterX = 0;
let pendingCenterY = 0;

canvas.addEventListener(
  'pointerdown',
  (e) => {
    dragging = true;

    dragStartX = e.clientX;
    dragStartY = e.clientY;

    pendingCenterX = view.centerX;
    pendingCenterY = view.centerY;

    canvas.setPointerCapture(
      e.pointerId
    );
  }
);

canvas.addEventListener(
  'pointermove',
  (e) => {
    if (!dragging) {
      return;
    }

    const rect =
      canvas.getBoundingClientRect();

    const pixelDx =
      e.clientX - dragStartX;

    const pixelDy =
      e.clientY - dragStartY;

    canvas.style.transform =
      `translate(${pixelDx}px, ${pixelDy}px)`;

    const dx =
      pixelDx /
      rect.height *
      2.0 *
      view.scale;

    const dy =
      pixelDy /
      rect.height *
      2.0 *
      view.scale;

    pendingCenterX =
      view.centerX - dx;

    pendingCenterY =
      view.centerY + dy;
  }
);

canvas.addEventListener(
  'pointerup',
  (e) => {
    if (!dragging) {
      return;
    }

    dragging = false;

    canvas.releasePointerCapture(
      e.pointerId
    );

    canvas.style.transform = 'none';

    view.centerX =
      pendingCenterX;

    view.centerY =
      pendingCenterY;

    resetAndStart();
  }
);

let wheelDebounce = null;

canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();

    const rect =
      canvas.getBoundingClientRect();

    const aspect =
      rect.width / rect.height;

    const u =
      ((e.clientX - rect.left) /
        rect.width -
        0.5) *
      2.0 *
      aspect;

    const v =
      ((e.clientY - rect.top) /
        rect.height -
        0.5) *
      2.0;

    const zBeforeX =
      view.centerX +
      u * view.scale;

    const zBeforeY =
      view.centerY -
      v * view.scale;

    const zoomFactor =
      e.deltaY < 0
        ? 0.85
        : 1 / 0.85;

    view.scale *=
      zoomFactor;

    view.centerX =
      zBeforeX -
      u * view.scale;

    view.centerY =
      zBeforeY +
      v * view.scale;

    clearTimeout(
      wheelDebounce
    );

    wheelDebounce =
      setTimeout(
        resetAndStart,
        120
      );
  },
  {
    passive: false
  }
);

recomputeBtn.addEventListener(
  'click',
  () => {
    view.centerX =
      parseFloat(
        centerXInput.value
      );

    view.centerY =
      parseFloat(
        centerYInput.value
      );

    view.scale =
      parseFloat(
        scaleInput.value
      );

    resetAndStart();
  }
);

resetViewBtn.addEventListener(
  'click',
  () => {
    view = {
      ...DEFAULT_VIEW
    };

    resetAndStart();
  }
);

resolutionSelect.addEventListener(
  'change',
  resetAndStart
);

window.addEventListener(
  'resize',
  () => {
    if (!device) {
      return;
    }

    resetAndStart();
  }
);

gradientAlgorithm.value =
  gradient.algorithm;

gradientLengthInput.value =
  gradient.length;

renderGradientStops();
drawGradientPreview();

(async function boot() {
  try {
    await initGPU();
    resetAndStart();
  } catch (err) {
    console.error(err);
  }
})();
