struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0) var<uniform> dims: vec2<u32>; // [m, n]
@group(0) @binding(1) var<storage, read> grad: Matrix;      // size m*n
@group(0) @binding(2) var<storage, read_write> biasGrad: Matrix; // size n

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = dims.y;
    let m = dims.x;

    let idx = gid.x;
    if (idx >= n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < m; i = i + 1u) {
        sum = sum + grad.data[i * n + idx];
    }

    biasGrad.data[idx] = sum;
}
