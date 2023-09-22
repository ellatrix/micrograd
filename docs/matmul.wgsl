struct Meta {
    M: u32, // Rows of A
    N: u32, // Columns of B
    K: u32, // Columns of A and rows of B, which is the depth of the matrix multiplication
}

@group(0) @binding(0) var<uniform> uniforms: Meta;
@group(0) @binding(1) var<storage,read_write> c: array<f32>;
@group(0) @binding(2) var<storage,read> a: array<f32>;
@group(0) @binding(3) var<storage,read> b: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var M: u32 = uniforms.M;
    var N: u32 = uniforms.N;
    var K: u32 = uniforms.K;
    var x: u32 = global_id.x;
    var y: u32 = global_id.y;
    var sum: f32 = 0.0;

    // Check if we are out of bounds.
    if (x >= N || y >= M) {
        return;
    }

    for (var k: u32 = 0u; k < K; k = k + 1u) {
        sum += a[y * K + k] * b[k * N + x];
    }

    c[x + y * N] = sum;
}