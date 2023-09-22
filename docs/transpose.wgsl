struct Meta {
    M: u32, // Rows of A
    N: u32, // Columns of A
}

@group(0) @binding(0) var<uniform> uniforms: Meta;
@group(0) @binding(1) var<storage,read_write> c: array<f32>;
@group(0) @binding(2) var<storage,read> a: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var M: u32 = uniforms.M;
    var N: u32 = uniforms.N;
    var x: u32 = global_id.x;
    var y: u32 = global_id.y;

    // Check if we are out of bounds.
    if (x >= N || y >= M) {
        return;
    }

    // Transpose logic: Swap row and column indices
    c[x * M + y] = a[y * M + x];
}