// Row-wise LayerNorm over the last dimension of a row-major [m, h] tensor:
// y = (x - mean) / sqrt(var + eps) * gamma + beta
//
// One invocation per row (portable; row lengths are model hidden/FFN sizes).

// wgpu caps a dispatch dimension at 65535 workgroups, so grids too wide for
// one dimension are folded into y (see `grid_1d`). Rebuild the flat index.

struct Dims {
    m: u32,
    h: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ngroups: vec3<u32>) {
    let row = gid.x + gid.y * ngroups.x * 64u;
    if (row >= dims.m) {
        return;
    }
    let base = row * dims.h;
    var sum = 0.0;
    for (var j = 0u; j < dims.h; j = j + 1u) {
        sum = sum + x[base + j];
    }
    let mean = sum / f32(dims.h);
    var varsum = 0.0;
    for (var j = 0u; j < dims.h; j = j + 1u) {
        let d = x[base + j] - mean;
        varsum = varsum + d * d;
    }
    let inv = 1.0 / sqrt(varsum / f32(dims.h) + dims.eps);
    for (var j = 0u; j < dims.h; j = j + 1u) {
        y[base + j] = (x[base + j] - mean) * inv * gamma[j] + beta[j];
    }
}
