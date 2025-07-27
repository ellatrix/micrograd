struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0) var<uniform> dims: vec2<u32>; // [B, T]
@group(0) @binding(1) var<storage, read_write> matrix: Matrix;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B = dims.x;
    let T = dims.y;
    let b = gid.z;
    let row = gid.y;
    let col = gid.x;

    if (b >= B || row >= T || col >= T) {
        return;
    }

    let base = (b * T + row) * T;
    let idx = base + col;

    // Step 1: Mask upper triangle
    if (col > row) {
        matrix.data[idx] = 0.0;
        return;
    }

    // Step 2: Find max over valid entries in row
    var maxVal: f32 = f32(-3.4e38);
    for (var k: u32 = 0u; k <= row; k = k + 1u) {
        let v = matrix.data[base + k];
        if (v > maxVal) {
            maxVal = v;
        }
    }

    // Step 3: Compute denominator
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k <= row; k = k + 1u) {
        sum = sum + exp(matrix.data[base + k] - maxVal);
    }

    // Step 4: Normalize
    matrix.data[idx] = exp(matrix.data[idx] - maxVal) / sum;
}
