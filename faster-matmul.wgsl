struct Meta {
  M: u32,
  N: u32,
  ND4: u32,
  KD4: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Meta;
@group(0) @binding(1) var<storage, read_write> array_c: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> array_a: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> array_b: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> array_bias: array<vec4<f32>>;

fn store_result(x: u32, y: u32, ND4: u32, c0: vec4<f32>, c1: vec4<f32>) {
  array_c[x + 0u + y * ND4] = c0;
  array_c[x + 1u + y * ND4] = c1;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let M = uniforms.M;
  let N = uniforms.N;
  let ND4 = uniforms.ND4;
  let KD4 = uniforms.KD4;

  let x = global_id.x;
  let y = global_id.y;

  // Each thread handles a 4x2 tile in the output matrix
  if (x * 8u >= N || y * 4u >= M) {
    return;
  }

  let row0 = y * 4u + 0u;
  let row1 = y * 4u + 1u;
  let row2 = y * 4u + 2u;
  let row3 = y * 4u + 3u;
  let col0 = x * 2u + 0u;
  let col1 = x * 2u + 1u;

  var sum00 = vec4<f32>();
  var sum01 = vec4<f32>();
  var sum02 = vec4<f32>();
  var sum03 = vec4<f32>();
  var sum10 = vec4<f32>();
  var sum11 = vec4<f32>();
  var sum12 = vec4<f32>();
  var sum13 = vec4<f32>();

  for (var k: u32 = 0u; k < KD4; k = k + 1u) {
    let a0 = array_a[row0 * KD4 + k];
    let a1 = array_a[row1 * KD4 + k];
    let a2 = array_a[row2 * KD4 + k];
    let a3 = array_a[row3 * KD4 + k];

    // Channel 0
    var brow = array_b[(k * 4u + 0u) * ND4 + col0];
    sum00 += vec4<f32>(a0.x) * brow;
    sum01 += vec4<f32>(a1.x) * brow;
    sum02 += vec4<f32>(a2.x) * brow;
    sum03 += vec4<f32>(a3.x) * brow;

    brow = array_b[(k * 4u + 0u) * ND4 + col1];
    sum10 += vec4<f32>(a0.x) * brow;
    sum11 += vec4<f32>(a1.x) * brow;
    sum12 += vec4<f32>(a2.x) * brow;
    sum13 += vec4<f32>(a3.x) * brow;

    // Channel 1
    brow = array_b[(k * 4u + 1u) * ND4 + col0];
    sum00 += vec4<f32>(a0.y) * brow;
    sum01 += vec4<f32>(a1.y) * brow;
    sum02 += vec4<f32>(a2.y) * brow;
    sum03 += vec4<f32>(a3.y) * brow;

    brow = array_b[(k * 4u + 1u) * ND4 + col1];
    sum10 += vec4<f32>(a0.y) * brow;
    sum11 += vec4<f32>(a1.y) * brow;
    sum12 += vec4<f32>(a2.y) * brow;
    sum13 += vec4<f32>(a3.y) * brow;

    // Channel 2
    brow = array_b[(k * 4u + 2u) * ND4 + col0];
    sum00 += vec4<f32>(a0.z) * brow;
    sum01 += vec4<f32>(a1.z) * brow;
    sum02 += vec4<f32>(a2.z) * brow;
    sum03 += vec4<f32>(a3.z) * brow;

    brow = array_b[(k * 4u + 2u) * ND4 + col1];
    sum10 += vec4<f32>(a0.z) * brow;
    sum11 += vec4<f32>(a1.z) * brow;
    sum12 += vec4<f32>(a2.z) * brow;
    sum13 += vec4<f32>(a3.z) * brow;

    // Channel 3
    brow = array_b[(k * 4u + 3u) * ND4 + col0];
    sum00 += vec4<f32>(a0.w) * brow;
    sum01 += vec4<f32>(a1.w) * brow;
    sum02 += vec4<f32>(a2.w) * brow;
    sum03 += vec4<f32>(a3.w) * brow;

    brow = array_b[(k * 4u + 3u) * ND4 + col1];
    sum10 += vec4<f32>(a0.w) * brow;
    sum11 += vec4<f32>(a1.w) * brow;
    sum12 += vec4<f32>(a2.w) * brow;
    sum13 += vec4<f32>(a3.w) * brow;
  }

  // Add bias
  let bias0 = array_bias[col0];
  let bias1 = array_bias[col1];

  sum00 += bias0;
  sum01 += bias0;
  sum02 += bias0;
  sum03 += bias0;

  sum10 += bias1;
  sum11 += bias1;
  sum12 += bias1;
  sum13 += bias1;

  // Write results
  if (row0 < M) { store_result(col0, row0, ND4, sum00, sum10); }
  if (row1 < M) { store_result(col0, row1, ND4, sum01, sum11); }
  if (row2 < M) { store_result(col0, row2, ND4, sum02, sum12); }
  if (row3 < M) { store_result(col0, row3, ND4, sum03, sum13); }
}
