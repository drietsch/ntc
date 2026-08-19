// Per-head scaled dot-product attention scores with key masking.
//
// q: [lq, h], k: [lkv, h] (heads packed along the feature dim, head_dim =
// h / heads). Output scores: [n_heads * lq, lkv] row-major, where row
// (local_head * lq + i) holds head `head_base + local_head`, query
// position `i`.
//
// `head_base` / `n_heads` let one dispatch cover a SUBSET of heads: the full
// score tensor is heads x lq x lkv floats, which exceeds WebGPU's 128 MiB
// storage-binding limit for long fusion sequences, so the caller chunks.
//
// Masked key positions (kv_mask[j] == 0) get -3.0e38: like the reference's
// f32::MIN, it underflows to exactly 0 after softmax.

struct Dims {
    lq: u32,
    lkv: u32,
    h: u32,
    heads: u32,
    scale: f32,
    head_base: u32,
    n_heads: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> kv_mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

const MASKED: f32 = -3.0e38;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let row = gid.y;
    if (row >= dims.n_heads * dims.lq || j >= dims.lkv) {
        return;
    }
    if (kv_mask[j] == 0u) {
        scores[row * dims.lkv + j] = MASKED;
        return;
    }
    let head = dims.head_base + row / dims.lq;
    let i = row % dims.lq;
    let hd = dims.h / dims.heads;
    let off = head * hd;
    var acc = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
        acc = acc + q[i * dims.h + off + d] * k[j * dims.h + off + d];
    }
    scores[row * dims.lkv + j] = acc * dims.scale;
}
