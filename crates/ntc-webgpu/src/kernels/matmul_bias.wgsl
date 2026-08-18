// C[m, n] = A[m, k] * B[k, n] (+ bias[n] when dims.has_bias != 0).
//
// Row-major f32. Accumulation is sequential over k, matching the reference
// `ntc_model::ops::linear` accumulation order (parity within f32 tolerance;
// backends may contract to FMA).

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    var acc = 0.0;
    for (var kk = 0u; kk < dims.k; kk = kk + 1u) {
        acc = acc + a[row * dims.k + kk] * b[kk * dims.n + col];
    }
    if (dims.has_bias != 0u) {
        acc = acc + bias[col];
    }
    c[row * dims.n + col] = acc;
}
