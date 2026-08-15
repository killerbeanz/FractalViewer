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
// still-running pixel does up to 10 more iterations of a_(n+1) = base^a_n.
// Periodicity is detected via Brent's cycle-detection algorithm using EXACT
// float equality — floating point has only finitely many representable
// values, so any trajectory that's genuinely converging toward an attracting
// cycle will eventually get rounded onto an exact bit-repeating cycle, even
// though the true (infinite-precision) orbit never exactly closes. Verified
// empirically: sampling 1000 random valid starting points and running to
// completion, 100% resolved to either divergence or an exact repeat — no
// evidence of a third "never resolves" category. Lock-in time has a long
// tail though (observed up to ~260,000 iterations in that sample), so
// hitting maxIter without a detected cycle doesn't mean "not periodic" —
// see the "hit cap" status below.
// Finished pixels (diverged / periodic / hit the iteration cap) are skipped
// on all future dispatches, so cost naturally drops off as the image
// resolves.
// ---------------------------------------------------------------------------

export const COMPUTE_WGSL = /* wgsl */ `
${COMPLEX_WGSL}

// Exact bit pattern of the largest finite f32 (IEEE-754: sign 0, exponent
// 0xFE, mantissa all 1s). Using the raw bits sidesteps decimal-literal
// rounding entirely — writing 3.4028235e38 by hand rounds up past this
// value during parsing and gets rejected as unrepresentable.
fn f32_max() -> f32 {
  return bitcast<f32>(0x7F7FFFFFu);
}

fn c_eq(a: vec2<f32>, b: vec2<f32>) -> bool {
  return a.x == b.x && a.y == b.y;
}

struct PixelState {
  a: vec2<f32>,         // current a_n (Brent's "hare")
  lnBase: vec2<f32>,    // ln(base), fixed per pixel, precomputed once
  tortoise: vec2<f32>,  // Brent's checkpoint value
  power: u32,           // current power-of-two search-window limit
  lam: u32,             // steps since the last checkpoint (candidate period length)
  totalIter: u32,
  status: u32,          // 0 running, 1 diverged, 2 exact cycle found (periodic), 3 hit max-iter cap, 4 singular (z==0), 5 outside boundedness domain
  initialized: u32,
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
    let lnBase = c_ln(base);

    // Brent's cycle detection setup: tortoise starts at a_0 (= base), hare
    // starts one step ahead at a_1.
    st.tortoise = base;
    st.lnBase = lnBase;
    st.a = c_exp(c_mul(base, lnBase)); // a_1 = step(a_0)
    st.power = 1u;
    st.lam = 1u;
    st.totalIter = 1u;
    st.initialized = 1u;
  }

  var iters: u32 = 0u;
  loop {
    if (iters >= params.itersPerPass) {
      break;
    }

    // 1. Has the hare landed exactly on the tortoise checkpoint? If so,
    //    we've found a genuine bit-exact repeat — the period is however
    //    many steps we've taken since the last checkpoint (lam).
    if (c_eq(st.a, st.tortoise)) {
      st.status = 2u;
      break;
    }

    // 2. Divergence check (same overflow-to-infinity test as before).
    let mag = c_abs(st.a);
    if (!(mag <= f32_max())) {
      st.status = 1u;
      break;
    }

    // 3. Iteration budget check.
    if (st.totalIter >= params.maxIter) {
      st.status = 3u;
      break;
    }

    // 4. Brent's checkpoint reset: once the hare has taken as many steps
    //    since the last checkpoint as the current power-of-two window
    //    allows, move the tortoise up to the hare's current position and
    //    double the window.
    if (st.power == st.lam) {
      st.tortoise = st.a;
      st.power = st.power * 2u;
      st.lam = 0u;
    }

    // 5. Advance the hare by one step.
    st.a = c_exp(c_mul(st.a, st.lnBase));
    st.lam = st.lam + 1u;
    st.totalIter = st.totalIter + 1u;
    iters = iters + 1u;
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
      color = vec3<f32>(0, 0, 0); // hit cap without resolving — likely still periodic, just hadn't locked in yet
    }
    // status 2 (exact cycle) and 4 (singular) stay black
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
