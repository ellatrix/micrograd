struct Meta {
    M: u32, // Rows of A
    N: u32, // Columns of B
    K: u32, // Columns of A and rows of B, which is the depth of the matrix multiplication
    aT: u32, // transpose flag for A (0=no, 1=yes)
    bT: u32, // transpose flag for B (0=no, 1=yes)
}

@group(0) @binding(0) var<uniform> uniforms: Meta;
@group(0) @binding(1) var<storage,read_write> c: array<f32>;
@group(0) @binding(2) var<storage,read> a: array<f32>;
@group(0) @binding(3) var<storage,read> b: array<f32>;
@group(0) @binding(4) var<storage,read> bias: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M: u32 = uniforms.M;
    let N: u32 = uniforms.N;
    let K: u32 = uniforms.K;
    let aT: u32 = uniforms.aT;
    let bT: u32 = uniforms.bT;
    let x: u32 = global_id.x;
    let y: u32 = global_id.y;

    // Check if we are out of bounds.
    if (x >= N || y >= M) {
        return;
    }

    var sum: f32 = bias[x];

    for (var k: u32 = 0u; k < K; k = k + 1u) {
        var a_index = y * K + k;
        var b_index = k * N + x;
        if (aT == 1) {
            a_index = k * M + y;
        }
        if (bT == 1) {
            b_index = x * K + k;
        }
        sum += a[a_index] * b[b_index];
    }

    c[x + y * N] = sum;
}
