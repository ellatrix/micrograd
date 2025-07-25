struct Meta {
    M: u32,
    N: u32,
    K: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Meta;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;

const tileSize: u32 = 16;

var<workgroup> tileA: array<array<f32, tileSize>, tileSize>;
var<workgroup> tileB: array<array<f32, tileSize>, tileSize>;

@compute @workgroup_size(tileSize, tileSize)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let M = uniforms.M;
    let N = uniforms.N;
    let K = uniforms.K;

    let x = global_id.x;
    let y = global_id.y;
    let lx = local_id.x;
    let ly = local_id.y;

    var acc: f32 = 0.0;
    let numTiles = (K + tileSize - 1u) / tileSize;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let tiledK = t * tileSize;

        // Load tileA
        if (y < M && (tiledK + lx) < K) {
            tileA[ly][lx] = a[y * K + (tiledK + lx)];
        } else {
            tileA[ly][lx] = 0.0;
        }

        // Load tileB
        if ((tiledK + ly) < K && x < N) {
            tileB[ly][lx] = b[(tiledK + ly) * N + x];
        } else {
            tileB[ly][lx] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < tileSize; k = k + 1u) {
            acc = acc + tileA[ly][k] * tileB[k][lx];
        }

        workgroupBarrier();
    }

    if (x < N && y < M) {
        c[y * N + x] = acc;
    }
}
