struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0) var<uniform> dims: vec2<u32>; // [B, T]
@group(0) @binding(1) var<storage, read> out: Matrix;
@group(0) @binding(2) var<storage, read> dout: Matrix;
@group(0) @binding(3) var<storage, read_write> din: Matrix;

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

    if (col > row) {
        // Upper triangle remains zero
        let idx = (b * T + row) * T + col;
        din.data[idx] = 0.0;
        return;
    }

    let base = (b * T + row) * T;
    var sum: f32 = 0.0;

    for (var k: u32 = 0u; k <= row; k++) {
        let out_k = out.data[base + k];
        let out_col = out.data[base + col];
        let delta = select(0.0, 1.0, k == col);
        let term = out_k * (delta - out_col) * dout.data[base + k];
        sum = sum + term;
    }

    din.data[base + col] = sum;
}
