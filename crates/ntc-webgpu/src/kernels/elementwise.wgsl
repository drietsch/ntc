// Elementwise kernels: exact-erf GELU (in place) and residual add (in place).
//
// The erf approximation is Abramowitz-Stegun 7.1.26 with the SAME constants
// as `ntc_model::ops` (the normative CPU reference).

// wgpu caps a dispatch dimension at 65535 workgroups, so grids too wide for
// one dimension are folded into y (see `grid_1d`). Rebuild the flat index.

struct Dims {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> other: array<f32>;

fn erf_as(v: f32) -> f32 {
    let sign = select(1.0, -1.0, v < 0.0);
    let av = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * av);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * exp(-av * av);
    return sign * y;
}

// GELU(v) = 0.5 * v * (1 + erf(v / sqrt(2)))
@compute @workgroup_size(64, 1, 1)
fn gelu_main(@builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(num_workgroups) ngroups: vec3<u32>) {
    let i = gid.x + gid.y * ngroups.x * 64u;
    if (i >= dims.n) {
        return;
    }
    let v = data[i];
    data[i] = 0.5 * v * (1.0 + erf_as(v / 1.4142135623730951));
}

// data[i] += other[i]
@compute @workgroup_size(64, 1, 1)
fn add_main(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(num_workgroups) ngroups: vec3<u32>) {
    let i = gid.x + gid.y * ngroups.x * 64u;
    if (i >= dims.n) {
        return;
    }
    data[i] = data[i] + other[i];
}
