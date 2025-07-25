struct Meta {
    M: u32, // Rows of A
    N: u32, // Columns of B
    K: u32, // Columns of A and rows of B
    aT: u32, // transpose flag for A (0=no, 1=yes)
    bT: u32, // transpose flag for B (0=no, 1=yes)
}

@group(0) @binding(0) var<uniform> uniforms: Meta;
@group(0) @binding(1) var<storage,read_write> c: array<f32>;
@group(0) @binding(2) var<storage,read> a: array<f32>;
@group(0) @binding(3) var<storage,read> b: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = uniforms.M;
    let N = uniforms.N;
    let K = uniforms.K;
    let aT = uniforms.aT;
    let bT = uniforms.bT;

    let x = global_id.x;
    let y = global_id.y;

    if (x >= N || y >= M) {
        return;
    }

    var sum: f32 = 0.0;

    for (var k: u32 = 0u; k < K; k = k + 1u) {
        var a_index: u32;
        if (aT == 1u) {
            a_index = y + k * M;
        } else {
            a_index = y * K + k;
        }

        var b_index: u32;
        if (bT == 1u) {
            b_index = x * K + k;
        } else {
            b_index = k * N + x;
        }

        sum = sum + a[a_index] * b[b_index];
    }

    c[y * N + x] = sum;
}
