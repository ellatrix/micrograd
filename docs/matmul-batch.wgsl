struct Matrix {
  data: array<f32>,
};
struct Meta {
    B: u32,
    M: u32,
    N: u32,
    K: u32,
    aT: u32,
    bT: u32,
};
@group(0) @binding(0) var<uniform> dims: Meta;
@group(0) @binding(1) var<storage, read> lhs: Matrix;
@group(0) @binding(2) var<storage, read> rhs: Matrix;
@group(0) @binding(3) var<storage, read_write> result: Matrix;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = dims.B;
    let M = dims.M;
    let N = dims.N;
    let K = dims.K;
    let aT = dims.aT;
    let bT = dims.bT;

    let b = gid.z;
    let i = gid.y;
    let j = gid.x;

    if (b >= B || i >= M || j >= N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        var a_index = ((b * M + i) * K + k);
        var b_index = ((b * K + k) * N + j);

        if (aT == 1) {
            a_index = ((b * K + k) * M + i);
        }
        if (bT == 1) {
            b_index = ((b * N + j) * K + k);
        }

        sum = sum + lhs.data[a_index] * rhs.data[b_index];
    }

    let out_index = ((b * M + i) * N + j);
    result.data[out_index] = sum;
}
