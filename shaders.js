export const COMPLEX_WGSL = /* wgsl */ `
fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a + b;
}

fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a - b;
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

fn c_reciprocal(a: vec2<f32>) -> vec2<f32> {
  let denom = a.x * a.x + a.y * a.y;

  return vec2<f32>(
    a.x / denom,
    -a.y / denom
  );
}

fn c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return c_mul(
    a,
    c_reciprocal(b)
  );
}

fn c_scale(a: vec2<f32>, s: f32) -> vec2<f32> {
  return vec2<f32>(
    a.x * s,
    a.y * s
  );
}

fn c_abs(a: vec2<f32>) -> f32 {
  return length(a);
}

fn c_ln(a: vec2<f32>) -> vec2<f32> {
  let r = length(a);
  let theta = atan2(a.y, a.x);

  return vec2<f32>(
    log(max(r, 1e-30)),
    theta
  );
}

fn c_exp(a: vec2<f32>) -> vec2<f32> {
  let r = exp(a.x);

  return vec2<f32>(
    r * cos(a.y),
    r * sin(a.y)
  );
}

fn c_lambertw0(x: vec2<f32>) -> vec2<f32> {
  var w: vec2<f32>;

  if (length(x) < 1.0) {
    w = x;
  } else {
    let lx = c_ln(x);
    w = c_sub(
      lx,
      c_ln(lx)
    );
  }

  for (
    var i: u32 = 0u;
    i < 30u;
    i = i + 1u
  ) {
    let ew =
      c_exp(w);

    let wew =
      c_mul(
        w,
        ew
      );

    let f =
      c_sub(
        wew,
        x
      );

    let wp1 =
      vec2<f32>(
        w.x + 1.0,
        w.y
      );

    let wp2 =
      vec2<f32>(
        w.x + 2.0,
        w.y
      );

    let denomA =
      c_mul(
        ew,
        wp1
      );

    let denomB =
      c_div(
        c_mul(
          wp2,
          f
        ),
        c_scale(
          wp1,
          2.0
        )
      );

    let denom =
      c_sub(
        denomA,
        denomB
      );

    w =
      c_sub(
        w,
        c_div(
          f,
          denom
        )
      );
  }

  return w;
}
`;

export const COMPUTE_WGSL = /* wgsl */ `
${COMPLEX_WGSL}

fn f32_max() -> f32 {
  return bitcast<f32>(
    0x7F7FFFFFu
  );
}

fn c_eq(
  a: vec2<f32>,
  b: vec2<f32>
) -> bool {
  return
    a.x == b.x &&
    a.y == b.y;
}

struct PixelState {
  a: vec2<f32>,
  lnBase: vec2<f32>,
  tortoise: vec2<f32>,
  power: u32,
  lam: u32,
  totalIter: u32,
  status: u32,
  initialized: u32,
};

struct Params {
  width: u32,
  height: u32,
  maxIter: u32,
  itersPerPass: u32,
  centerX: f32,
  centerY: f32,
  scale: f32,
  _pad: f32,
};

struct GradientData {
  colors: array<vec4<f32>, 16>,
  weights: array<f32, 16>,
  count: u32,
  algorithm: u32,
  length: f32,
  _pad: f32,
};

@group(0) @binding(0)
var<storage, read_write>
states: array<PixelState>;

@group(0) @binding(1)
var outputTex:
texture_storage_2d<
  rgba8unorm,
  write
>;

@group(0) @binding(2)
var<uniform>
params: Params;

@group(0) @binding(3)
var<storage, read_write>
doneCounter:
atomic<u32>;

@group(0) @binding(4)
var<storage, read>
gradient: GradientData;

fn gradientEase(t: f32) -> f32 {
  if (gradient.algorithm == 1u) {
    return
      t *
      t *
      (3.0 - 2.0 * t);
  }

  if (gradient.algorithm == 2u) {
    return
      t *
      t *
      t *
      (
        t *
        (t * 6.0 - 15.0) +
        10.0
      );
  }

  if (gradient.algorithm == 3u) {
    return
      (
        1.0 -
        cos(
          3.14159265 * t
        )
      ) *
      0.5;
  }

  return t;
}

fn palette(iteration: f32) -> vec3<f32> {
  if (gradient.count == 0u) {
    return vec3<f32>(0.0);
  }

  if (gradient.count == 1u) {
    return gradient.colors[0].xyz;
  }

  let position =
    fract(
      iteration /
      max(
        gradient.length,
        1.0
      )
    );

  var totalWeight = 0.0;

  for (
    var i: u32 = 0u;
    i < gradient.count;
    i = i + 1u
  ) {
    totalWeight +=
      max(
        gradient.weights[i],
        0.0001
      );
  }

  let distance =
    position *
    totalWeight;

  var accumulated = 0.0;

  for (
    var i: u32 = 0u;
    i < gradient.count;
    i = i + 1u
  ) {
    let weight =
      max(
        gradient.weights[i],
        0.0001
      );

    let nextAccumulated =
      accumulated +
      weight;

    if (
      distance <
        nextAccumulated ||
      i ==
        gradient.count - 1u
    ) {
      let localT =
        (
          distance -
          accumulated
        ) /
        weight;

      let easedT =
        gradientEase(
          clamp(
            localT,
            0.0,
            1.0
          )
        );

      let nextIndex =
        (
          i + 1u
        ) %
        gradient.count;

      let a =
        gradient.colors[i].xyz;

      let b =
        gradient.colors[nextIndex].xyz;

      return
        a +
        (
          b - a
        ) *
        easedT;
    }

    accumulated =
      nextAccumulated;
  }

  return gradient.colors[0].xyz;
}

@compute
@workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id)
  gid: vec3<u32>
) {
  if (
    gid.x >= params.width ||
    gid.y >= params.height
  ) {
    return;
  }

  let idx =
    gid.y *
    params.width +
    gid.x;

  var st =
    states[idx];

  if (st.status != 0u) {
    return;
  }

  let px =
    vec2<i32>(
      i32(gid.x),
      i32(gid.y)
    );

  if (st.initialized == 0u) {
    let aspect =
      f32(params.width) /
      f32(params.height);

    let u =
      (
        f32(gid.x) /
        f32(params.width) -
        0.5
      ) *
      2.0 *
      aspect;

    let v =
      (
        f32(gid.y) /
        f32(params.height) -
        0.5
      ) *
      2.0;

    let z =
      vec2<f32>(
        params.centerX +
        u * params.scale,

        params.centerY -
        v * params.scale
      );

    if (length(z) < 1e-8) {
      st.status =
        4u;

      st.initialized =
        1u;

      states[idx] =
        st;

      atomicAdd(
        &doneCounter,
        1u
      );

      textureStore(
        outputTex,
        px,
        vec4<f32>(
          0.0,
          0.0,
          0.0,
          1.0
        )
      );

      return;
    }

    let lnz =
      c_ln(z);

    let wArg =
      c_div(
        vec2<f32>(
          -lnz.x,
          -lnz.y
        ),
        z
      );

    let w =
      c_lambertw0(
        wArg
      );

    let negZ =
      vec2<f32>(
        -z.x,
        -z.y
      );

    let boundVal =
      length(
        c_ln(
          c_div(
            c_mul(
              negZ,
              w
            ),
            lnz
          )
        )
      );

    if (boundVal <= 1.0) {
      st.status =
        5u;

      st.initialized =
        1u;

      states[idx] =
        st;

      atomicAdd(
        &doneCounter,
        1u
      );

      textureStore(
        outputTex,
        px,
        vec4<f32>(
          0.0,
          0.0,
          0.0,
          1.0
        )
      );

      return;
    }

    let invz =
      c_reciprocal(z);

    let base =
      c_exp(
        c_mul(
          invz,
          lnz
        )
      );

    let lnBase =
      c_ln(base);

    st.tortoise =
      base;

    st.lnBase =
      lnBase;

    st.a =
      c_exp(
        c_mul(
          base,
          lnBase
        )
      );

    st.power =
      1u;

    st.lam =
      1u;

    st.totalIter =
      1u;

    st.initialized =
      1u;
  }

  var iters: u32 = 0u;

  loop {
    if (
      iters >=
      params.itersPerPass
    ) {
      break;
    }

    if (
      c_eq(
        st.a,
        st.tortoise
      )
    ) {
      st.status =
        2u;

      break;
    }

    let mag =
      c_abs(st.a);

    if (!(mag <= f32_max())) {
      st.status =
        1u;

      break;
    }

    if (
      st.totalIter >=
      params.maxIter
    ) {
      st.status =
        3u;

      break;
    }

    if (
      st.power ==
      st.lam
    ) {
      st.tortoise =
        st.a;

      st.power =
        st.power * 2u;

      st.lam =
        0u;
    }

    st.a =
      c_exp(
        c_mul(
          st.a,
          st.lnBase
        )
      );

    st.lam =
      st.lam + 1u;

    st.totalIter =
      st.totalIter + 1u;

    iters =
      iters + 1u;
  }

  states[idx] =
    st;

  if (st.status != 0u) {
    atomicAdd(
      &doneCounter,
      1u
    );

    var color =
      vec3<f32>(
        0.0,
        0.0,
        0.0
      );

    if (st.status == 1u) {
      color =
        palette(
          f32(
            st.totalIter
          )
        );
    }

    textureStore(
      outputTex,
      px,
      vec4<f32>(
        color,
        1.0
      )
    );
  }
}
`;

export const RENDER_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position)
  pos: vec4<f32>,

  @location(0)
  uv: vec2<f32>,
};

@vertex
fn vs_main(
  @builtin(vertex_index)
  vi: u32
) -> VSOut {
  var positions =
    array<vec2<f32>, 3>(
      vec2<f32>(
        -1.0,
        -1.0
      ),

      vec2<f32>(
        3.0,
        -1.0
      ),

      vec2<f32>(
        -1.0,
        3.0
      )
    );

  var out: VSOut;

  let p =
    positions[vi];

  out.pos =
    vec4<f32>(
      p,
      0.0,
      1.0
    );

  out.uv =
    vec2<f32>(
      (p.x + 1.0) *
      0.5,

      1.0 -
      (p.y + 1.0) *
      0.5
    );

  return out;
}

@group(0) @binding(0)
var tex:
texture_2d<f32>;

@group(0) @binding(1)
var samp:
sampler;

@fragment
fn fs_main(
  in: VSOut
) -> @location(0) vec4<f32> {
  return textureSample(
    tex,
    samp,
    in.uv
  );
}
`;
