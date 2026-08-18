// Numerically stable softmax over the last axis of a row-major [m, n]
// tensor, in place. One invocation per row. Mirrors
// `ntc_model::ops::softmax_rows` (max-subtract, exp, divide; a row whose
// exp-sum is 0 is left as the exp values, i.e. all zeros).

struct Dims {
    m: u32,
    n: u32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.m) {
        return;
    }
    let base = row * dims.n;
    var maxv = data[base];
    for (var j = 1u; j < dims.n; j = j + 1u) {
        maxv = max(maxv, data[base + j]);
    }
    var sum = 0.0;
    for (var j = 0u; j < dims.n; j = j + 1u) {
        let e = exp(data[base + j] - maxv);
        data[base + j] = e;
        sum = sum + e;
    }
    if (sum > 0.0) {
        for (var j = 0u; j < dims.n; j = j + 1u) {
            data[base + j] = data[base + j] / sum;
        }
    }
}
