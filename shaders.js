// All GPU code lives here as WGSL source strings. No build step needed —
// these get compiled by the browser's WebGPU implementation at runtime.

// ---------------------------------------------------------------------------
// Complex arithmetic helpers, shared by the compute kernel.
// Currently 32-bit float pairs (vec2<f32> = (re, im)). If you later move to
// arbitrary/extended precision, this is the one block that needs replacing —
// everything downstream just calls c_add/c_mul/c_exp/c_ln etc., so the
// iteration logic itself shouldn't need to change.
// ---------------------------------------------------------------------------
export const COMPLEX_WGSL = /* wgsl */ `
fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a + b;
}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b;
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn c_reciprocal(a: vec2<f32>) -> vec2<f32> {
  let denom = a.x * a.x + a.y * a.y;
  return vec2<f32>(a.x / denom, -a.y / denom);
}

fn c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return c_mul(a, c_reciprocal(b));
}

fn c_scale(a: vec2<f32>, s: f32) -> vec2<f32> {
  return vec2<f32>(a.x * s, a.y * s);
}

fn c_abs(a: vec2<f32>) -> f32 {
  return length(a);
}

// principal complex log: ln|a| + i*arg(a)
fn c_ln(a: vec2<f32>) -> vec2<f32> {
  let r = length(a);
  let theta = atan2(a.y, a.x);
  return vec2<f32>(log(max(r, 1e-30)), theta);
}

fn c_exp(a: vec2<f32>) -> vec2<f32> {
  let r = exp(a.x);
  return vec2<f32>(r * cos(a.y), r * sin(a.y));
}

// Principal branch (W0) of the complex Lambert W function: solves W*e^W = x.
// No closed form, so this refines an asymptotic starting guess with fixed
// Halley's-method iterations (cubically convergent, so this many iterations
// is generous away from the branch point at x = -1/e; near it, convergence
// is slower, but that's a thin region and errors there just shift which
// pixels near the boundary get gated).
fn c_lambertw0(x: vec2<f32>) -> vec2<f32> {
  var w: vec2<f32>;
  if (length(x) < 1.0) {
    w = x; // W(x) ~= x for small |x|
  } else {
    let lx = c_ln(x);
    w = c_sub(lx, c_ln(lx)); // W(x) ~= ln(x) - ln(ln(x)) for large |x|
  }

  for (var i: u32 = 0u; i < 30u; i = i + 1u) {
    let ew = c_exp(w);
    let wew = c_mul(w, ew);
    let f = c_sub(wew, x);
    let wp1 = vec2<f32>(w.x + 1.0, w.y);
    let wp2 = vec2<f32>(w.x + 2.0, w.y);
    let denomA = c_mul(ew, wp1);
    let denomB = c_div(c_mul(wp2, f), c_scale(wp1, 2.0));
    let denom = c_sub(denomA, denomB);
    w = c_sub(w, c_div(f, denom));
  }
  return w;
}
`;

// ---------------------------------------------------------------------------
// Compute kernel. One thread per pixel. Each dispatch = one "pass": every
// still-running pixel does up to 10 iterations of a_(n+1) = base^a_n, tracks
// the up/down run-length pattern of ln|a_n|, and checks that pattern list
// for a repeat. Finished pixels (diverged / periodic / hit the iteration
// cap) are skipped on all future dispatches, so cost naturally drops off as
// the image resolves.
// ---------------------------------------------------------------------------
const HISTORY_SIZE = 24; // run-lengths kept per pixel; tune for memory vs. how long a cycle can be

export const COMPUTE_WGSL = /* wgsl */ `
${COMPLEX_WGSL}

const HISTORY_SIZE: u32 = ${HISTORY_SIZE}u;
// Exact bit pattern of the largest finite f32 (IEEE-754: sign 0, exponent
// 0xFE, mantissa all 1s). Using the raw bits sidesteps decimal-literal
// rounding entirely — writing 3.4028235e38 by hand rounds up past this
// value during parsing and gets rejected as unrepresentable.
fn f32_max() -> f32 {
  return bitcast<f32>(0x7F7FFFFFu);
}

struct PixelState {
  a: vec2<f32>,        // current a_n
  lnBase: vec2<f32>,   // ln(base), fixed per pixel, precomputed once
  lastMag: f32,        // ln|a_(n-1)|, for direction comparison
  curDir: i32,         // -1 down, 0 unset, 1 up
  curRunLen: u32,
  historyLen: u32,
  totalIter: u32,
  status: u32,          // 0 running, 1 diverged, 2 periodic, 3 hit max-iter cap, 4 singular (z==0), 5 outside boundedness domain
  initialized: u32,
  history: array<u32, HISTORY_SIZE>,
};

struct Params {
  width: u32,
  height: u32,
  maxIter: u32,
  itersPerPass: u32,
  centerX: f32,
  centerY: f32,
  scale: f32,           // half-height of the view, in complex-plane units
  _pad: f32,             // keeps struct size a clean 32 bytes for the uniform binding
};

@group(0) @binding(0) var<storage, read_write> states: array<PixelState>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> doneCounter: atomic<u32>;

fn push_history(st: ptr<function, PixelState>, runLen: u32) {
  if ((*st).historyLen < HISTORY_SIZE) {
    (*st).history[(*st).historyLen] = runLen;
    (*st).historyLen = (*st).historyLen + 1u;
  } else {
    for (var i: u32 = 0u; i < HISTORY_SIZE - 1u; i = i + 1u) {
      (*st).history[i] = (*st).history[i + 1u];
    }
    (*st).history[HISTORY_SIZE - 1u] = runLen;
  }
}

// Looks for the pattern list becoming periodic: does the tail of the
// history repeat with some period p? Checks the smallest p first.
fn detect_periodicity(st: ptr<function, PixelState>) -> bool {
  let n = (*st).historyLen;
  let maxP = n / 2u;
  for (var p: u32 = 1u; p <= maxP; p = p + 1u) {
    var isMatch = true;
    for (var i: u32 = 0u; i < p; i = i + 1u) {
      let x = (*st).history[n - 1u - i];
      let y = (*st).history[n - 1u - i - p];
      if (x != y) {
        isMatch = false;
        break;
      }
    }
    if (isMatch) {
      return true;
    }
  }
  return false;
}

fn palette(t: f32) -> vec3<f32> {
  // smooth cyclical palette for "time to infinity" coloring
  let a = vec3<f32>(0.5, 0.5, 0.5);
  let b = vec3<f32>(0.5, 0.5, 0.5);
  let c = vec3<f32>(1.0, 1.0, 1.0);
  let d = vec3<f32>(0.30, 0.20, 0.55);
  return a + b * cos(6.28318 * (c * t + d));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let idx = gid.y * params.width + gid.x;
  var st = states[idx];

  if (st.status != 0u) {
    return; // already resolved on an earlier pass
  }

  let px = vec2<i32>(i32(gid.x), i32(gid.y));

  if (st.initialized == 0u) {
    let aspect = f32(params.width) / f32(params.height);
    let u = (f32(gid.x) / f32(params.width) - 0.5) * 2.0 * aspect;
    let v = (f32(gid.y) / f32(params.height) - 0.5) * 2.0;
    let z = vec2<f32>(params.centerX + u * params.scale, params.centerY - v * params.scale);

    if (length(z) < 1e-8) {
      // z == 0: 1/z and ln(z) are undefined. Mark and move on.
      st.status = 4u;
      st.initialized = 1u;
      states[idx] = st;
      atomicAdd(&doneCounter, 1u);
      textureStore(outputTex, px, vec4<f32>(0.0, 0.0, 0.0, 1.0));
      return;
    }

    let lnz = c_ln(z);

    // Boundedness gate: abs(ln(-z*W(-ln(z)/z)/ln(z))) > 1 is the region
    // where the tower is guaranteed to be periodic or divergent. Outside
    // it, skip the iteration entirely and paint black immediately.
    // If the Lambert W solve is numerically unstable here (e.g. near a
    // branch cut) boundVal comes out NaN, and "boundVal <= 1.0" is false
    // for NaN under IEEE754 — so we fall through to normal iteration
    // rather than wrongly forcing black.
    let wArg = c_div(vec2<f32>(-lnz.x, -lnz.y), z);
    let w = c_lambertw0(wArg);
    let negZ = vec2<f32>(-z.x, -z.y);
    let boundVal = length(c_ln(c_div(c_mul(negZ, w), lnz)));

    if (boundVal <= 1.0) {
      st.status = 5u;
      st.initialized = 1u;
      states[idx] = st;
      atomicAdd(&doneCounter, 1u);
      textureStore(outputTex, px, vec4<f32>(0.0, 0.0, 0.0, 1.0));
      return;
    }

    let invz = c_reciprocal(z);
    let base = c_exp(c_mul(invz, lnz)); // base = z^(1/z)

    st.a = base;                 // a_0 = base
    st.lnBase = c_ln(base);
    st.lastMag = log(max(c_abs(base), 1e-30));
    st.curDir = 0;
    st.curRunLen = 0u;
    st.historyLen = 0u;
    st.totalIter = 0u;
    st.initialized = 1u;
  }

  var iters: u32 = 0u;
  loop {
    if (iters >= params.itersPerPass) {
      break;
    }
    if (st.totalIter >= params.maxIter) {
      st.status = 3u;
      break;
    }

    let a_new = c_exp(c_mul(st.a, st.lnBase)); // a_(n+1) = base^a_n
    let mag = c_abs(a_new);

    // No try/catch on a GPU — the equivalent is checking whether the
    // float math actually overflowed to IEEE infinity, rather than
    // picking an arbitrary escape magnitude. f32_max() is the largest
    // finite f32; anything strictly above it can only be +inf (or NaN,
    // which shows up here from indeterminate forms — e.g. inf*0 — that
    // arise mid-overflow). Either way the sequence has genuinely blown
    // up, so both count as divergence. Comparisons against NaN are
    // false under IEEE754, so "!(mag <= f32_max())" catches both cases.
    if (!(mag <= f32_max())) {
      st.status = 1u;
      break;
    }

    let L_new = log(max(mag, 1e-30));
    let delta = L_new - st.lastMag;
    let EPS = 1e-6;
    var dir: i32 = st.curDir;
    if (delta > EPS) {
      dir = 1;
    } else if (delta < -EPS) {
      dir = -1;
    }

    if (st.curDir == 0) {
      st.curDir = dir;
      st.curRunLen = 1u;
    } else if (dir == st.curDir) {
      st.curRunLen = st.curRunLen + 1u;
    } else {
      push_history(&st, st.curRunLen);
      st.curDir = dir;
      st.curRunLen = 1u;
    }

    st.lastMag = L_new;
    st.a = a_new;
    st.totalIter = st.totalIter + 1u;
    iters = iters + 1u;
  }

  if (st.status == 0u && st.historyLen >= 4u) {
    if (detect_periodicity(&st)) {
      st.status = 2u;
    }
  }

  states[idx] = st;

  if (st.status != 0u) {
    atomicAdd(&doneCounter, 1u);
    var color = vec3<f32>(0.0, 0.0, 0.0);
    if (st.status == 1u) {
      // diverged: color by time-to-infinity
      let t = log(f32(st.totalIter) + 1.0) / log(f32(params.maxIter) + 1.0);
      color = palette(t);
    } else if (st.status == 3u) {
      color = vec3<f32>(0.12, 0.12, 0.14); // hit cap without resolving — likely periodic, pattern too long to catch
    }
    // status 2 (periodic) and 4 (singular) stay black
    textureStore(outputTex, px, vec4<f32>(color, 1.0));
  }
}
`;

// ---------------------------------------------------------------------------
// Fullscreen blit: draws the output texture (written by the compute pass)
// onto the canvas.
// ---------------------------------------------------------------------------
export const RENDER_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  var out: VSOut;
  let p = positions[vi];
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
  return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, in.uv);
}
`;