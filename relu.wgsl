struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0) var<uniform> length: u32;
@group(0) @binding(1) var<storage, read_write> matrix: Matrix;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= length) {
        return;
    }

    let x = matrix.data[idx];
    matrix.data[idx] = max(0.0, x);
}
