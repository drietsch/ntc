// Attention context: y[i, d] = sum_j P[head(d)][i, j] * V[j, d]
//
// p: [heads * lq, lkv] softmaxed scores (row (head * lq + i)),
// v: [lkv, h] with heads packed along the feature dim,
// y: [lq, h] — per-head contexts land concatenated in their feature slice.

struct Dims {
    lq: u32,
    lkv: u32,
    h: u32,
    heads: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let i = gid.y;
    if (d >= dims.h || i >= dims.lq) {
        return;
    }
    let hd = dims.h / dims.heads;
    let head = d / hd;
    let pbase = (head * dims.lq + i) * dims.lkv;
    var acc = 0.0;
    for (var j = 0u; j < dims.lkv; j = j + 1u) {
        acc = acc + p[pbase + j] * v[j * dims.h + d];
    }
    y[i * dims.h + d] = acc;
}
