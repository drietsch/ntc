//! Naive f32 reference operations.
//!
//! These are the **normative numerics** for NTC-Web inference: every GPU
//! kernel is parity-tested against them. Accumulation order is fixed
//! (sequential over the reduction dimension) so results are bit-stable across
//! platforms. Clarity beats speed here on purpose.

// Index-style loops are deliberate here: they mirror the WGSL kernels 1:1.
#![allow(clippy::needless_range_loop)]

use crate::tensor::Tensor;

/// `y[m, n] = x[m, k] · w[k, n] (+ bias[n])`
pub fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Tensor {
    assert_eq!(x.rank(), 2, "linear: x must be rank 2");
    assert_eq!(w.rank(), 2, "linear: w must be rank 2");
    let (m, k) = (x.shape[0], x.shape[1]);
    let (wk, n) = (w.shape[0], w.shape[1]);
    assert_eq!(k, wk, "linear: inner dims {k} vs {wk}");
    let mut out = Tensor::zeros(&[m, n]);
    for i in 0..m {
        let xi = &x.data[i * k..(i + 1) * k];
        let oi = &mut out.data[i * n..(i + 1) * n];
        for (kk, &xv) in xi.iter().enumerate() {
            if xv == 0.0 {
                continue;
            }
            let wr = &w.data[kk * n..(kk + 1) * n];
            for j in 0..n {
                oi[j] += xv * wr[j];
            }
        }
        if let Some(b) = bias {
            for j in 0..n {
                oi[j] += b.data[j];
            }
        }
    }
    out
}

/// Row-wise LayerNorm over the last dimension of a rank-2 tensor.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let (m, h) = (x.shape[0], x.shape[1]);
    let mut out = Tensor::zeros(&[m, h]);
    for i in 0..m {
        let row = &x.data[i * h..(i + 1) * h];
        let mean = row.iter().sum::<f32>() / h as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
        let inv = 1.0 / (var + eps).sqrt();
        let o = &mut out.data[i * h..(i + 1) * h];
        for j in 0..h {
            o[j] = (row[j] - mean) * inv * weight.data[j] + bias.data[j];
        }
    }
    out
}

/// Exact (erf-based) GELU, matching PyTorch `nn.GELU()` default.
pub fn gelu(x: &mut Tensor) {
    for v in &mut x.data {
        *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
    }
}

/// Abramowitz–Stegun 7.1.26 erf approximation (max abs error ~1.5e-7,
/// far below cross-backend f32 tolerance).
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

pub fn add_inplace(x: &mut Tensor, y: &Tensor) {
    assert_eq!(x.shape, y.shape);
    for (a, b) in x.data.iter_mut().zip(&y.data) {
        *a += b;
    }
}

/// Numerically stable softmax over the last axis of a rank-2 tensor,
/// honoring an additive mask (`-inf` style: masked entries get `f32::MIN`).
pub fn softmax_rows(x: &mut Tensor) {
    let (m, n) = (x.shape[0], x.shape[1]);
    for i in 0..m {
        let row = &mut x.data[i * n..(i + 1) * n];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Multi-head attention: `q_states` attends over `kv_states`.
/// `kv_mask[j] == false` masks key/value position `j`.
/// Shapes: q_states `[Lq, H]`, kv_states `[Lkv, H]`; weights per the naming
/// contract; `heads * head_dim == H`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q_states: &Tensor,
    kv_states: &Tensor,
    kv_mask: &[bool],
    wq: (&Tensor, &Tensor),
    wk: (&Tensor, &Tensor),
    wv: (&Tensor, &Tensor),
    wo: (&Tensor, &Tensor),
    heads: usize,
) -> Tensor {
    let (lq, h) = (q_states.shape[0], q_states.shape[1]);
    let lkv = kv_states.shape[0];
    assert_eq!(kv_mask.len(), lkv);
    let hd = h / heads;
    let scale = 1.0 / (hd as f32).sqrt();

    let q = linear(q_states, wq.0, Some(wq.1));
    let k = linear(kv_states, wk.0, Some(wk.1));
    let v = linear(kv_states, wv.0, Some(wv.1));

    let mut ctx = Tensor::zeros(&[lq, h]);
    for head in 0..heads {
        let off = head * hd;
        // scores[lq, lkv]
        let mut scores = Tensor::zeros(&[lq, lkv]);
        for i in 0..lq {
            let qi = &q.data[i * h + off..i * h + off + hd];
            for j in 0..lkv {
                if !kv_mask[j] {
                    scores.data[i * lkv + j] = f32::MIN;
                    continue;
                }
                let kj = &k.data[j * h + off..j * h + off + hd];
                let mut dot = 0.0;
                for d in 0..hd {
                    dot += qi[d] * kj[d];
                }
                scores.data[i * lkv + j] = dot * scale;
            }
        }
        softmax_rows(&mut scores);
        for i in 0..lq {
            let ci = &mut ctx.data[i * h + off..i * h + off + hd];
            for j in 0..lkv {
                let p = scores.data[i * lkv + j];
                if p == 0.0 {
                    continue;
                }
                let vj = &v.data[j * h + off..j * h + off + hd];
                for d in 0..hd {
                    ci[d] += p * vj[d];
                }
            }
        }
    }
    linear(&ctx, wo.0, Some(wo.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_matches_manual() {
        let x = Tensor::from_vec(&[1, 3], vec![1.0, 2.0, 3.0]);
        let w = Tensor::from_vec(&[3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let b = Tensor::from_vec(&[2], vec![0.5, -0.5]);
        let y = linear(&x, &w, Some(&b));
        assert_eq!(y.data, vec![1.0 + 3.0 + 0.5, 2.0 + 3.0 - 0.5]);
    }

    #[test]
    fn layernorm_zero_mean_unit_var() {
        let x = Tensor::from_vec(&[1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let w = Tensor::from_vec(&[4], vec![1.0; 4]);
        let b = Tensor::from_vec(&[4], vec![0.0; 4]);
        let y = layer_norm(&x, &w, &b, 1e-5);
        let mean: f32 = y.data.iter().sum::<f32>() / 4.0;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn gelu_reference_points() {
        let mut x = Tensor::from_vec(&[1, 3], vec![-1.0, 0.0, 1.0]);
        gelu(&mut x);
        assert_relative_eq!(x.data[0], -0.158_655_25, epsilon = 1e-5);
        assert_relative_eq!(x.data[1], 0.0, epsilon = 1e-7);
        assert_relative_eq!(x.data[2], 0.841_344_8, epsilon = 1e-5);
    }

    #[test]
    fn softmax_masked_rows_sum_to_one() {
        let mut x = Tensor::from_vec(&[1, 3], vec![1.0, f32::MIN, 2.0]);
        softmax_rows(&mut x);
        assert_relative_eq!(x.data.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        assert_eq!(x.data[1], 0.0);
    }
}
