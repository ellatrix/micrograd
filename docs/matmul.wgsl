struct Meta {
    M: u32, // Rows of A
    N: u32, // Columns of B
    ND4: u32, // Columns of B 
    KD4: u32, // Columns of A and rows of B, which is the depth of the matrix multiplication
}

@group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
@group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;
@group(1) @binding(2) var<storage,read> array_bias: array<vec4<f32>>;

@group(0) @binding(0) var<uniform> uniforms: Meta;
@group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var M: u32 = uniforms.M;
    var N: u32 = uniforms.N;
    var ND4: u32 = uniforms.ND4;
    var KD4: u32 = uniforms.KD4;
    var x: u32 = global_id.x;
    var y: u32 = global_id.y;

    if (x * 8 >= N || y * 4 >= M) {
        return;
    }

    var sum00: vec4<f32> = vec4<f32>();
    var sum01: vec4<f32> = vec4<f32>();
    var sum02: vec4<f32> = vec4<f32>();
    var sum03: vec4<f32> = vec4<f32>();
    var sum10: vec4<f32> = vec4<f32>();
    var sum11: vec4<f32> = vec4<f32>();
    var sum12: vec4<f32> = vec4<f32>();
    var sum13: vec4<f32> = vec4<f32>();

    for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
    }

    var array_bias_1: vec4<f32> = array_bias[x * 2u + 0u];
    sum00 = sum00 + array_bias_1;
    sum01 = sum01 + array_bias_1;
    sum02 = sum02 + array_bias_1;
    sum03 = sum03 + array_bias_1;

    var array_bias_2: vec4<f32> = array_bias[x * 2u + 1u];
    sum10 = sum10 + array_bias_2;
    sum11 = sum11 + array_bias_2;
    sum12 = sum12 + array_bias_2;
    sum13 = sum13 + array_bias_2;

    array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
    array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
    array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
    array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
    array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
    array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
    array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
    array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
}