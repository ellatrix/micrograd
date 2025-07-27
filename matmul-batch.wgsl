struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0) var<uniform> dims: vec4<u32>; // B, M, N, K
@group(0) @binding(1) var<storage, read> lhs: Matrix;
@group(0) @binding(2) var<storage, read> rhs: Matrix;
@group(0) @binding(3) var<storage, read_write> result: Matrix;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = dims.x;
    let M = dims.y;
    let N = dims.z;
    let K = dims.w;

    let b = gid.z;
    let i = gid.y;
    let j = gid.x;

    if (b >= B || i >= M || j >= N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        // --- Change these lines based on aT/bT ---
        // let a_index = ((b * M + i) * K + k); // aT = 0
        // let a_index = ((b * K + k) * M + i); // aT = 1

        // let b_index = ((b * K + k) * N + j); // bT = 0
        // let b_index = ((b * N + j) * K + k); // bT = 1

        sum = sum + lhs.data[/*INDEXA*/] * rhs.data[/*INDEXB*/];
    }

    let out_index = ((b * M + i) * N + j);
    result.data[out_index] = sum;
}
