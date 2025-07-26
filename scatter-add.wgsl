@group(0) @binding(0) var<uniform> uniforms: vec2<u32>; // [dim, numRows]
@group(0) @binding(1) var<storage, read_write> output: array<atomic<i32>>; // shape: [M * dim]
@group(0) @binding(2) var<storage, read> grad: array<f32>;                 // shape: [numRows * dim]
@group(0) @binding(3) var<storage, read> indices: array<i32>;             // shape: [numRows]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let dim = uniforms.x;
    let numRows = uniforms.y;

    if (row >= numRows) {
        return;
    }

    let dstIndex = indices[row];

    for (var j: u32 = 0u; j < dim; j = j + 1u) {
        let srcOffset = row * dim + j;
        let dstOffset = u32(dstIndex) * dim + j;

        let addVal = grad[srcOffset];
        loop {
            let oldBits = atomicLoad(&output[dstOffset]);
            let oldVal = bitcast<f32>(oldBits);
            let newVal = oldVal + addVal;
            let newBits = bitcast<i32>(newVal);
            let result = atomicCompareExchangeWeak(&output[dstOffset], oldBits, newBits);
            if (result.exchanged) {
                break;
            }
        }
    }
}
