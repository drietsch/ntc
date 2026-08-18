//! GPU op executor: compiled compute pipelines, a transient-buffer arena,
//! recording helpers for each kernel, and blocking one-shot host wrappers
//! (used by the kernel parity tests).
//!
//! Correctness first: every dispatch runs in its own compute pass, dims go
//! through a fresh uniform buffer per dispatch, and transient storage
//! buffers come from a size-keyed free list that is only recycled after the
//! submission has completed (`read_back` waits before reclaiming).

use std::collections::HashMap;

use ntc_core::NtcError;
use ntc_model::Tensor;
use wgpu::util::DeviceExt;

use crate::device::{GpuCaps, WgpuContext};
use crate::kernels;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDims {
    m: u32,
    k: u32,
    n: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormDims {
    m: u32,
    h: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElementwiseDims {
    n: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScoreDims {
    lq: u32,
    lkv: u32,
    h: u32,
    heads: u32,
    scale: f32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxDims {
    m: u32,
    n: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnCtxDims {
    lq: u32,
    lkv: u32,
    h: u32,
    heads: u32,
}

struct Kernels {
    matmul: wgpu::ComputePipeline,
    layernorm: wgpu::ComputePipeline,
    gelu: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    attn_scores: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    attn_ctx: wgpu::ComputePipeline,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Self {
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let pipeline = |label: &str, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let matmul = module("matmul_bias", kernels::MATMUL_BIAS);
        let layernorm = module("layernorm", kernels::LAYERNORM);
        let elementwise = module("elementwise", kernels::ELEMENTWISE);
        let attn_scores = module("attn_scores", kernels::ATTN_SCORES);
        let softmax = module("softmax", kernels::SOFTMAX);
        let attn_ctx = module("attn_ctx", kernels::ATTN_CTX);
        Self {
            matmul: pipeline("matmul_bias", &matmul, "main"),
            layernorm: pipeline("layernorm", &layernorm, "main"),
            gelu: pipeline("gelu", &elementwise, "gelu_main"),
            add: pipeline("add", &elementwise, "add_main"),
            attn_scores: pipeline("attn_scores", &attn_scores, "main"),
            softmax: pipeline("softmax", &softmax, "main"),
            attn_ctx: pipeline("attn_ctx", &attn_ctx, "main"),
        }
    }
}

/// Size-keyed free list of transient storage buffers. Buffers move to
/// `in_use` on alloc and back to `free` on `reclaim()`, which callers only
/// invoke after the submission using them has completed.
#[derive(Default)]
struct BufferArena {
    free: HashMap<u64, Vec<wgpu::Buffer>>,
    in_use: Vec<wgpu::Buffer>,
}

impl BufferArena {
    fn alloc(&mut self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        let buf = self
            .free
            .get_mut(&size)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            });
        self.in_use.push(buf.clone());
        buf
    }

    fn reclaim(&mut self) {
        for buf in self.in_use.drain(..) {
            self.free.entry(buf.size()).or_default().push(buf);
        }
    }
}

/// Records kernel dispatches into command encoders and reads results back.
pub struct GpuExecutor {
    ctx: WgpuContext,
    kernels: Kernels,
    arena: BufferArena,
    /// Bound in the bias slot when a matmul has no bias (never read).
    dummy: wgpu::Buffer,
}

impl GpuExecutor {
    pub fn new(ctx: WgpuContext) -> Self {
        let kernels = Kernels::new(&ctx.device);
        let dummy = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy-bias"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            ctx,
            kernels,
            arena: BufferArena::default(),
            dummy,
        }
    }

    pub fn caps(&self) -> &GpuCaps {
        &self.ctx.caps
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.ctx.device
    }

    pub(crate) fn create_encoder(&self) -> wgpu::CommandEncoder {
        self.ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ntc") })
    }

    /// Transient storage buffer for `floats` f32 elements.
    pub(crate) fn alloc(&mut self, floats: usize) -> wgpu::Buffer {
        self.arena
            .alloc(&self.ctx.device, (floats.max(1) * 4) as u64)
    }

    pub(crate) fn upload_f32(&mut self, data: &[f32]) -> wgpu::Buffer {
        let buf = self.alloc(data.len());
        self.ctx
            .queue
            .write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    pub(crate) fn upload_mask(&mut self, mask: &[bool]) -> wgpu::Buffer {
        let words: Vec<u32> = mask.iter().map(|&b| u32::from(b)).collect();
        let buf = self.alloc(words.len());
        self.ctx
            .queue
            .write_buffer(&buf, 0, bytemuck::cast_slice(&words));
        buf
    }

    pub(crate) fn write_f32(&self, buf: &wgpu::Buffer, offset_bytes: u64, data: &[f32]) {
        self.ctx
            .queue
            .write_buffer(buf, offset_bytes, bytemuck::cast_slice(data));
    }

    fn uniform<T: bytemuck::Pod>(&self, value: &T) -> wgpu::Buffer {
        self.ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// One dispatch in its own compute pass; `buffers` bind at 0..len.
    fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        groups: (u32, u32, u32),
    ) {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }

    /// `C[m, n] = A[m, k] · W[k, n] (+ bias)`; returns the output buffer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_matmul(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        bias: Option<&wgpu::Buffer>,
        m: usize,
        k: usize,
        n: usize,
    ) -> wgpu::Buffer {
        let out = self.alloc(m * n);
        let dims = self.uniform(&MatmulDims {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            has_bias: u32::from(bias.is_some()),
        });
        let bias_buf = bias.unwrap_or(&self.dummy);
        self.dispatch(
            enc,
            &self.kernels.matmul,
            &[&dims, a, w, bias_buf, &out],
            (div_ceil(n, 8), div_ceil(m, 8), 1),
        );
        out
    }

    /// Row-wise LayerNorm of `x` `[m, h]`; returns the output buffer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_layernorm(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        gamma: &wgpu::Buffer,
        beta: &wgpu::Buffer,
        m: usize,
        h: usize,
        eps: f32,
    ) -> wgpu::Buffer {
        let out = self.alloc(m * h);
        let dims = self.uniform(&LayerNormDims {
            m: m as u32,
            h: h as u32,
            eps,
            _pad: 0,
        });
        self.dispatch(
            enc,
            &self.kernels.layernorm,
            &[&dims, x, gamma, beta, &out],
            (div_ceil(m, 64), 1, 1),
        );
        out
    }

    /// In-place exact-erf GELU over `n` elements.
    pub(crate) fn record_gelu(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        n: usize,
    ) {
        let dims = self.uniform(&ElementwiseDims {
            n: n as u32,
            _pad: [0; 3],
        });
        self.dispatch(
            enc,
            &self.kernels.gelu,
            &[&dims, x],
            (div_ceil(n, 64), 1, 1),
        );
    }

    /// In-place residual add: `x += y` over `n` elements.
    pub(crate) fn record_add(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: usize,
    ) {
        let dims = self.uniform(&ElementwiseDims {
            n: n as u32,
            _pad: [0; 3],
        });
        self.dispatch(
            enc,
            &self.kernels.add,
            &[&dims, x, y],
            (div_ceil(n, 64), 1, 1),
        );
    }

    /// Masked per-head `QK^T · scale`; returns scores `[heads * lq, lkv]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attn_scores(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        kv_mask: &wgpu::Buffer,
        lq: usize,
        lkv: usize,
        h: usize,
        heads: usize,
    ) -> wgpu::Buffer {
        let out = self.alloc(heads * lq * lkv);
        let head_dim = h / heads;
        let dims = self.uniform(&AttnScoreDims {
            lq: lq as u32,
            lkv: lkv as u32,
            h: h as u32,
            heads: heads as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            _pad: [0; 3],
        });
        self.dispatch(
            enc,
            &self.kernels.attn_scores,
            &[&dims, q, k, kv_mask, &out],
            (div_ceil(lkv, 8), div_ceil(heads * lq, 8), 1),
        );
        out
    }

    /// In-place row softmax of `x` `[m, n]`.
    pub(crate) fn record_softmax(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        m: usize,
        n: usize,
    ) {
        let dims = self.uniform(&SoftmaxDims {
            m: m as u32,
            n: n as u32,
            _pad: [0; 2],
        });
        self.dispatch(
            enc,
            &self.kernels.softmax,
            &[&dims, x],
            (div_ceil(m, 64), 1, 1),
        );
    }

    /// `P · V` per head, heads concatenated; returns context `[lq, h]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attn_ctx(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        p: &wgpu::Buffer,
        v: &wgpu::Buffer,
        lq: usize,
        lkv: usize,
        h: usize,
        heads: usize,
    ) -> wgpu::Buffer {
        let out = self.alloc(lq * h);
        let dims = self.uniform(&AttnCtxDims {
            lq: lq as u32,
            lkv: lkv as u32,
            h: h as u32,
            heads: heads as u32,
        });
        self.dispatch(
            enc,
            &self.kernels.attn_ctx,
            &[&dims, p, v, &out],
            (div_ceil(h, 8), div_ceil(lq, 8), 1),
        );
        out
    }

    /// Full multi-head attention (`q_states` attends over `kv_states`) with
    /// projection weights, mirroring `ntc_model::ops::attention`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_attention(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        q_states: &wgpu::Buffer,
        kv_states: &wgpu::Buffer,
        kv_mask: &wgpu::Buffer,
        wq: (&wgpu::Buffer, &wgpu::Buffer),
        wk: (&wgpu::Buffer, &wgpu::Buffer),
        wv: (&wgpu::Buffer, &wgpu::Buffer),
        wo: (&wgpu::Buffer, &wgpu::Buffer),
        lq: usize,
        lkv: usize,
        h: usize,
        heads: usize,
    ) -> wgpu::Buffer {
        let q = self.record_matmul(enc, q_states, wq.0, Some(wq.1), lq, h, h);
        let k = self.record_matmul(enc, kv_states, wk.0, Some(wk.1), lkv, h, h);
        let v = self.record_matmul(enc, kv_states, wv.0, Some(wv.1), lkv, h, h);
        let scores = self.record_attn_scores(enc, &q, &k, kv_mask, lq, lkv, h, heads);
        self.record_softmax(enc, &scores, heads * lq, lkv);
        let ctx = self.record_attn_ctx(enc, &scores, &v, lq, lkv, h, heads);
        self.record_matmul(enc, &ctx, wo.0, Some(wo.1), lq, h, h)
    }

    /// Submit the encoder, copy each `(buffer, float_count)` source into a
    /// staging buffer, block until mapped, and return the host data.
    /// Reclaims the transient arena afterwards (queue is idle by then).
    pub(crate) fn read_back(
        &mut self,
        mut enc: wgpu::CommandEncoder,
        srcs: &[(&wgpu::Buffer, usize)],
    ) -> Result<Vec<Vec<f32>>, NtcError> {
        let stagings: Vec<wgpu::Buffer> = srcs
            .iter()
            .map(|&(src, floats)| {
                let size = (floats.max(1) * 4) as u64;
                let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
                staging
            })
            .collect();

        self.ctx.queue.submit(Some(enc.finish()));

        let mut results = Vec::with_capacity(stagings.len());
        for staging in &stagings {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.ctx.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|_| NtcError::Inference("wgpu map_async callback dropped".into()))?
                .map_err(|e| NtcError::Inference(format!("wgpu buffer map failed: {e:?}")))?;
            let data = slice.get_mapped_range();
            results.push(bytemuck::cast_slice::<u8, f32>(&data).to_vec());
            drop(data);
            staging.unmap();
        }
        self.arena.reclaim();
        Ok(results)
    }

    // ------------------------------------------------------------------
    // Blocking one-shot host wrappers (kernel parity tests).
    // ------------------------------------------------------------------

    /// GPU `ops::linear`: `x[m, k] · w[k, n] (+ bias)`.
    pub fn matmul_host(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, NtcError> {
        let (m, k) = (x.shape[0], x.shape[1]);
        let n = w.shape[1];
        let x_buf = self.upload_f32(&x.data);
        let w_buf = self.upload_f32(&w.data);
        let bias_buf = bias.map(|b| self.upload_f32(&b.data));
        let mut enc = self.create_encoder();
        let out = self.record_matmul(&mut enc, &x_buf, &w_buf, bias_buf.as_ref(), m, k, n);
        let mut res = self.read_back(enc, &[(&out, m * n)])?;
        Ok(Tensor::from_vec(&[m, n], res.remove(0)))
    }

    /// GPU `ops::layer_norm`.
    pub fn layer_norm_host(
        &mut self,
        x: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        eps: f32,
    ) -> Result<Tensor, NtcError> {
        let (m, h) = (x.shape[0], x.shape[1]);
        let x_buf = self.upload_f32(&x.data);
        let g_buf = self.upload_f32(&gamma.data);
        let b_buf = self.upload_f32(&beta.data);
        let mut enc = self.create_encoder();
        let out = self.record_layernorm(&mut enc, &x_buf, &g_buf, &b_buf, m, h, eps);
        let mut res = self.read_back(enc, &[(&out, m * h)])?;
        Ok(Tensor::from_vec(&[m, h], res.remove(0)))
    }

    /// GPU `ops::gelu` (returns the transformed tensor).
    pub fn gelu_host(&mut self, x: &Tensor) -> Result<Tensor, NtcError> {
        let x_buf = self.upload_f32(&x.data);
        let mut enc = self.create_encoder();
        self.record_gelu(&mut enc, &x_buf, x.data.len());
        let mut res = self.read_back(enc, &[(&x_buf, x.data.len())])?;
        Ok(Tensor::from_vec(&x.shape, res.remove(0)))
    }

    /// GPU `ops::add_inplace` (returns `x + y`).
    pub fn add_host(&mut self, x: &Tensor, y: &Tensor) -> Result<Tensor, NtcError> {
        let x_buf = self.upload_f32(&x.data);
        let y_buf = self.upload_f32(&y.data);
        let mut enc = self.create_encoder();
        self.record_add(&mut enc, &x_buf, &y_buf, x.data.len());
        let mut res = self.read_back(enc, &[(&x_buf, x.data.len())])?;
        Ok(Tensor::from_vec(&x.shape, res.remove(0)))
    }

    /// GPU `ops::softmax_rows` (returns the softmaxed tensor).
    pub fn softmax_host(&mut self, x: &Tensor) -> Result<Tensor, NtcError> {
        let (m, n) = (x.shape[0], x.shape[1]);
        let x_buf = self.upload_f32(&x.data);
        let mut enc = self.create_encoder();
        self.record_softmax(&mut enc, &x_buf, m, n);
        let mut res = self.read_back(enc, &[(&x_buf, m * n)])?;
        Ok(Tensor::from_vec(&[m, n], res.remove(0)))
    }

    /// GPU `ops::attention` (q/k/v/o projections + scores + softmax + ctx).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_host(
        &mut self,
        q_states: &Tensor,
        kv_states: &Tensor,
        kv_mask: &[bool],
        wq: (&Tensor, &Tensor),
        wk: (&Tensor, &Tensor),
        wv: (&Tensor, &Tensor),
        wo: (&Tensor, &Tensor),
        heads: usize,
    ) -> Result<Tensor, NtcError> {
        let (lq, h) = (q_states.shape[0], q_states.shape[1]);
        let lkv = kv_states.shape[0];
        let q_buf = self.upload_f32(&q_states.data);
        let kv_buf = self.upload_f32(&kv_states.data);
        let mask_buf = self.upload_mask(kv_mask);
        let up = |ex: &mut Self, t: (&Tensor, &Tensor)| {
            let w = ex.upload_f32(&t.0.data);
            let b = ex.upload_f32(&t.1.data);
            (w, b)
        };
        let (wq_w, wq_b) = up(self, wq);
        let (wk_w, wk_b) = up(self, wk);
        let (wv_w, wv_b) = up(self, wv);
        let (wo_w, wo_b) = up(self, wo);
        let mut enc = self.create_encoder();
        let out = self.record_attention(
            &mut enc,
            &q_buf,
            &kv_buf,
            &mask_buf,
            (&wq_w, &wq_b),
            (&wk_w, &wk_b),
            (&wv_w, &wv_b),
            (&wo_w, &wo_b),
            lq,
            lkv,
            h,
            heads,
        );
        let mut res = self.read_back(enc, &[(&out, lq * h)])?;
        Ok(Tensor::from_vec(&[lq, h], res.remove(0)))
    }
}

fn div_ceil(n: usize, d: usize) -> u32 {
    (n.div_ceil(d)) as u32
}
