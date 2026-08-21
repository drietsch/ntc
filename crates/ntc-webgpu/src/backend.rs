//! `WgpuBackend`: the V1 WebGPU inference backend.
//!
//! Split of work (accepted V1 layout, matches the project plan):
//! - CPU: embedding lookups/sums (utterance word+pos; schema word+pos+
//!   segment-kind+tool-index) and the head MLP/bilinear projections over the
//!   read-back states (tiny tensors; "head softmax/argmax on CPU").
//! - GPU: everything from the first LayerNorm onward — embedding norms, all
//!   encoder/schema transformer layers, and all fusion blocks (matmul+bias,
//!   masked per-head attention scores, softmax, context, output projection,
//!   residual add, LayerNorm, GELU).
//!
//! Weight strategy V1: one f32 storage buffer per transformer-resident
//! tensor, keyed by canonical name (per-matrix buffers, spec §36); the
//! embedding tables and head weights stay host-side.

use std::collections::HashMap;

use ntc_core::NtcError;
use ntc_model::backend::{Backend, HeadOutputs};
use ntc_model::inputs::{ModelInputs, ToolInput};
use ntc_model::weights::{tensor_specs, ModelWeights};
use ntc_model::{ops, NtcArchConfig, Tensor};
use wgpu::util::DeviceExt;

use crate::device::WgpuContext;
use crate::gpu::GpuExecutor;

/// Which tensors live in GPU storage buffers (everything the transformer
/// stack reads); the rest stay host-side for CPU embeddings/heads.
fn gpu_resident(name: &str) -> bool {
    name.starts_with("encoder.layer.")
        || name.starts_with("schema.layer.")
        || name.starts_with("fusion.block.")
        || name.starts_with("embeddings.norm.")
        || name.starts_with("schema.embeddings.norm.")
}

pub struct WgpuBackend {
    cfg: NtcArchConfig,
    exec: GpuExecutor,
    /// Per-tensor GPU weight buffers, keyed by canonical tensor name.
    wbufs: HashMap<String, wgpu::Buffer>,
    /// Host-side tensors: embedding tables, `fusion.no_tool.embedding`, heads.
    host: HashMap<String, Tensor>,
    /// Whether this model carries the head-codec v3 heads.
    weights_have_v3: bool,
    /// Whether this model carries the head-codec v4 filter-template head.
    weights_have_v4: bool,
}

impl WgpuBackend {
    /// Upload the transformer weights to the GPU and keep host copies of the
    /// embedding tables and head weights.
    pub fn new(
        cfg: NtcArchConfig,
        weights: &ModelWeights,
        ctx: WgpuContext,
    ) -> Result<Self, NtcError> {
        cfg.validate()?;
        weights.check(&cfg)?;
        let weights_have_v3 = weights.has_v3_heads();
        let weights_have_v4 = weights.has_filter_template_head();
        let exec = GpuExecutor::new(ctx);

        let mut wbufs = HashMap::new();
        let mut host = HashMap::new();
        // Head-codec v3 tensors are optional, so they are appended to the
        // manifest walk rather than folded into `tensor_specs`, which every
        // model must satisfy.
        let mut optional = if weights_have_v3 {
            ntc_model::weights::v3_head_specs(&cfg)
        } else {
            Vec::new()
        };
        // v4 is optional independently of v3: a model may declare a template
        // table without carrying the v3 heads.
        if weights_have_v4 {
            optional.extend(ntc_model::weights::v4_head_specs(&cfg));
        }
        let manifest = tensor_specs(&cfg).into_iter().chain(optional);
        for (name, _shape) in manifest {
            let tensor = weights.get(&name)?;
            if gpu_resident(&name) {
                let buf = exec
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&name),
                        contents: bytemuck::cast_slice(&tensor.data),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                wbufs.insert(name, buf);
            } else {
                host.insert(name, tensor.clone());
            }
        }

        Ok(Self {
            cfg,
            exec,
            wbufs,
            host,
            weights_have_v3,
            weights_have_v4,
        })
    }

    pub fn config(&self) -> &NtcArchConfig {
        &self.cfg
    }

    fn ht(&self, name: &str) -> Result<&Tensor, NtcError> {
        self.host
            .get(name)
            .ok_or_else(|| NtcError::Inference(format!("missing host tensor `{name}`")))
    }

    fn hwb(&self, prefix: &str) -> Result<(&Tensor, &Tensor), NtcError> {
        Ok((
            self.ht(&format!("{prefix}.weight"))?,
            self.ht(&format!("{prefix}.bias"))?,
        ))
    }

    /// Utterance embedding sum (word + position), pre-LayerNorm.
    fn embed_utterance_sum(&self, ids: &[u32]) -> Result<Vec<f32>, NtcError> {
        let h = self.cfg.hidden;
        let word = self.ht("embeddings.word.weight")?;
        let pos = self.ht("embeddings.position.weight")?;
        let mut x = vec![0.0f32; ids.len() * h];
        for (i, &id) in ids.iter().enumerate() {
            let wrow = word.row(id as usize);
            let prow = pos.row(i);
            for (j, o) in x[i * h..(i + 1) * h].iter_mut().enumerate() {
                *o = wrow[j] + prow[j];
            }
        }
        Ok(x)
    }

    /// Schema embedding sum (word + position + segment-kind + tool-index),
    /// pre-LayerNorm.
    fn embed_tool_sum(&self, tool: &ToolInput, tool_index: usize) -> Result<Vec<f32>, NtcError> {
        let h = self.cfg.hidden;
        let word = self.ht("embeddings.word.weight")?;
        let pos = self.ht("embeddings.position.weight")?;
        let kind_emb = self.ht("schema.embeddings.segment_kind.weight")?;
        let tool_emb = self.ht("schema.embeddings.tool_index.weight")?;
        let trow = tool_emb.row(tool_index);

        let ls = tool.ids.len();
        let mut x = vec![0.0f32; ls * h];
        for i in 0..ls {
            let wrow = word.row(tool.ids[i] as usize);
            let prow = pos.row(i);
            let krow = kind_emb.row(tool.kinds[i] as usize);
            for (j, o) in x[i * h..(i + 1) * h].iter_mut().enumerate() {
                *o = wrow[j] + prow[j] + krow[j] + trow[j];
            }
        }
        Ok(x)
    }

    // Head computations on CPU (identical numerics to `CpuRefBackend`).

    /// `dense -> gelu -> out` MLP head on a single state row; returns logits.
    fn mlp_head(&self, state: &[f32], dense: &str, out: &str) -> Result<Vec<f32>, NtcError> {
        let x = Tensor::from_vec(&[1, state.len()], state.to_vec());
        let (dw, db) = self.hwb(dense)?;
        let mut hidden = ops::linear(&x, dw, Some(db));
        ops::gelu(&mut hidden);
        let (ow, ob) = self.hwb(out)?;
        Ok(ops::linear(&hidden, ow, Some(ob)).data)
    }

    /// Plain linear head on a single state row.
    fn linear_head(&self, state: &[f32], prefix: &str) -> Result<Vec<f32>, NtcError> {
        let x = Tensor::from_vec(&[1, state.len()], state.to_vec());
        let (w, b) = self.hwb(prefix)?;
        Ok(ops::linear(&x, w, Some(b)).data)
    }

    /// Bilinear scores of one state against a set of states: `(s·W)·rows`.
    fn bilinear_scores(
        &self,
        state: &[f32],
        w_name: &str,
        targets: &Tensor,
        out: &mut [f32],
    ) -> Result<(), NtcError> {
        let w = self.ht(w_name)?;
        let x = Tensor::from_vec(&[1, state.len()], state.to_vec());
        let proj = ops::linear(&x, w, None);
        let h = state.len();
        for (j, o) in out.iter_mut().enumerate() {
            let row = &targets.data[j * h..(j + 1) * h];
            let mut dot = 0.0;
            for (p, r) in proj.data.iter().zip(row) {
                dot += p * r;
            }
            *o = dot;
        }
        Ok(())
    }
}

/// Look up a GPU weight buffer by canonical name.
fn wb<'a>(
    wbufs: &'a HashMap<String, wgpu::Buffer>,
    name: &str,
) -> Result<&'a wgpu::Buffer, NtcError> {
    wbufs
        .get(name)
        .ok_or_else(|| NtcError::Inference(format!("missing GPU weight buffer `{name}`")))
}

/// Attention with weights `{prefix}.{q,k,v,o}.{weight,bias}`.
#[allow(clippy::too_many_arguments)]
fn record_attention(
    exec: &mut GpuExecutor,
    wbufs: &HashMap<String, wgpu::Buffer>,
    enc: &mut wgpu::CommandEncoder,
    prefix: &str,
    q_states: &wgpu::Buffer,
    kv_states: &wgpu::Buffer,
    kv_mask: &wgpu::Buffer,
    lq: usize,
    lkv: usize,
    h: usize,
    heads: usize,
) -> Result<wgpu::Buffer, NtcError> {
    let pair = |p: &str| -> Result<(&wgpu::Buffer, &wgpu::Buffer), NtcError> {
        Ok((
            wb(wbufs, &format!("{prefix}.{p}.weight"))?,
            wb(wbufs, &format!("{prefix}.{p}.bias"))?,
        ))
    };
    let (wq, wk, wv, wo) = (pair("q")?, pair("k")?, pair("v")?, pair("o")?);
    Ok(exec.record_attention(
        enc, q_states, kv_states, kv_mask, wq, wk, wv, wo, lq, lkv, h, heads,
    ))
}

/// FFN sublayer (`up -> gelu -> down`) + residual + LayerNorm, with weights
/// `{prefix}.ffn.*`; returns the new states buffer.
#[allow(clippy::too_many_arguments)]
fn record_ffn_block(
    exec: &mut GpuExecutor,
    wbufs: &HashMap<String, wgpu::Buffer>,
    enc: &mut wgpu::CommandEncoder,
    prefix: &str,
    states: &wgpu::Buffer,
    len: usize,
    h: usize,
    f: usize,
    eps: f32,
) -> Result<wgpu::Buffer, NtcError> {
    let up = exec.record_matmul(
        enc,
        states,
        wb(wbufs, &format!("{prefix}.ffn.up.weight"))?,
        Some(wb(wbufs, &format!("{prefix}.ffn.up.bias"))?),
        len,
        h,
        f,
    );
    exec.record_gelu(enc, &up, len * f);
    let down = exec.record_matmul(
        enc,
        &up,
        wb(wbufs, &format!("{prefix}.ffn.down.weight"))?,
        Some(wb(wbufs, &format!("{prefix}.ffn.down.bias"))?),
        len,
        f,
        h,
    );
    exec.record_add(enc, states, &down, len * h);
    Ok(exec.record_layernorm(
        enc,
        states,
        wb(wbufs, &format!("{prefix}.ffn.norm.weight"))?,
        wb(wbufs, &format!("{prefix}.ffn.norm.bias"))?,
        len,
        h,
        eps,
    ))
}

/// One post-LN transformer layer (self-attn -> add+LN -> FFN -> add+LN).
#[allow(clippy::too_many_arguments)]
fn record_transformer_layer(
    exec: &mut GpuExecutor,
    wbufs: &HashMap<String, wgpu::Buffer>,
    enc: &mut wgpu::CommandEncoder,
    prefix: &str,
    states: wgpu::Buffer,
    mask: &wgpu::Buffer,
    len: usize,
    h: usize,
    f: usize,
    heads: usize,
    eps: f32,
) -> Result<wgpu::Buffer, NtcError> {
    let attn = record_attention(
        exec,
        wbufs,
        enc,
        &format!("{prefix}.attn"),
        &states,
        &states,
        mask,
        len,
        len,
        h,
        heads,
    )?;
    exec.record_add(enc, &states, &attn, len * h);
    let states = exec.record_layernorm(
        enc,
        &states,
        wb(wbufs, &format!("{prefix}.attn.norm.weight"))?,
        wb(wbufs, &format!("{prefix}.attn.norm.bias"))?,
        len,
        h,
        eps,
    );
    record_ffn_block(exec, wbufs, enc, prefix, &states, len, h, f, eps)
}

impl WgpuBackend {
    /// Record the full forward pass; returns the encoder plus the
    /// user/fused state buffers to read back.
    fn record_forward(
        &mut self,
        inputs: &ModelInputs,
    ) -> Result<(wgpu::CommandEncoder, wgpu::Buffer, wgpu::Buffer), NtcError> {
        let cfg = self.cfg.clone();
        let h = cfg.hidden;
        let f = cfg.ffn;
        let heads = cfg.heads;
        let eps = cfg.layer_norm_eps;
        let ls = cfg.max_schema_tokens;
        let lu = cfg.max_utterance_tokens;

        let n_tools = inputs.tools.len();

        // CPU embedding sums (pre-LayerNorm).
        let utt_sum = self.embed_utterance_sum(&inputs.utterance_ids)?;
        let tool_sums: Vec<Vec<f32>> = inputs
            .tools
            .iter()
            .enumerate()
            .map(|(t, tool)| self.embed_tool_sum(tool, t))
            .collect::<Result<_, _>>()?;
        let no_tool = self.ht("fusion.no_tool.embedding")?.data.clone();

        let exec = &mut self.exec;
        let wbufs = &self.wbufs;
        let mut enc = exec.create_encoder();

        // 1. Utterance encoder (GPU from the embedding LayerNorm onward).
        let utt_buf = exec.upload_f32(&utt_sum);
        let umask = exec.upload_mask(&inputs.utterance_mask);
        let mut user = exec.record_layernorm(
            &mut enc,
            &utt_buf,
            wb(wbufs, "embeddings.norm.weight")?,
            wb(wbufs, "embeddings.norm.bias")?,
            lu,
            h,
            eps,
        );
        for i in 0..cfg.encoder_layers {
            user = record_transformer_layer(
                exec,
                wbufs,
                &mut enc,
                &format!("encoder.layer.{i}"),
                user,
                &umask,
                lu,
                h,
                f,
                heads,
                eps,
            )?;
        }

        // 2. Schema encoders.
        let mut tool_bufs = Vec::with_capacity(n_tools);
        for (t, tool) in inputs.tools.iter().enumerate() {
            let x_buf = exec.upload_f32(&tool_sums[t]);
            let tmask = exec.upload_mask(&tool.mask);
            let mut states = exec.record_layernorm(
                &mut enc,
                &x_buf,
                wb(wbufs, "schema.embeddings.norm.weight")?,
                wb(wbufs, "schema.embeddings.norm.bias")?,
                ls,
                h,
                eps,
            );
            for i in 0..cfg.schema_layers {
                states = record_transformer_layer(
                    exec,
                    wbufs,
                    &mut enc,
                    &format!("schema.layer.{i}"),
                    states,
                    &tmask,
                    ls,
                    h,
                    f,
                    heads,
                    eps,
                )?;
            }
            tool_bufs.push(states);
        }

        // 3. Fusion over the packed sequence (+ NO_TOOL slot at the end).
        let s_len = n_tools * ls + 1;
        let packed = exec.alloc(s_len * h);
        for (t, tool_buf) in tool_bufs.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                tool_buf,
                0,
                &packed,
                (t * ls * h * 4) as u64,
                (ls * h * 4) as u64,
            );
        }
        exec.write_f32(&packed, ((s_len - 1) * h * 4) as u64, &no_tool);

        let mut packed_mask = Vec::with_capacity(s_len);
        for tool in &inputs.tools {
            packed_mask.extend_from_slice(&tool.mask);
        }
        packed_mask.push(true); // NO_TOOL slot
        let pmask = exec.upload_mask(&packed_mask);

        let mut fused = packed;
        for i in 0..cfg.fusion_blocks {
            let p = format!("fusion.block.{i}");
            let sa = record_attention(
                exec,
                wbufs,
                &mut enc,
                &format!("{p}.self"),
                &fused,
                &fused,
                &pmask,
                s_len,
                s_len,
                h,
                heads,
            )?;
            exec.record_add(&mut enc, &fused, &sa, s_len * h);
            fused = exec.record_layernorm(
                &mut enc,
                &fused,
                wb(wbufs, &format!("{p}.self.norm.weight"))?,
                wb(wbufs, &format!("{p}.self.norm.bias"))?,
                s_len,
                h,
                eps,
            );

            let ca = record_attention(
                exec,
                wbufs,
                &mut enc,
                &format!("{p}.cross"),
                &fused,
                &user,
                &umask,
                s_len,
                lu,
                h,
                heads,
            )?;
            exec.record_add(&mut enc, &fused, &ca, s_len * h);
            fused = exec.record_layernorm(
                &mut enc,
                &fused,
                wb(wbufs, &format!("{p}.cross.norm.weight"))?,
                wb(wbufs, &format!("{p}.cross.norm.bias"))?,
                s_len,
                h,
                eps,
            );

            fused = record_ffn_block(exec, wbufs, &mut enc, &p, &fused, s_len, h, f, eps)?;
        }

        Ok((enc, user, fused))
    }

    /// CPU head decode over read-back states (mirrors `CpuRefBackend`).
    fn decode_heads(
        &self,
        inputs: &ModelInputs,
        user_states: Tensor,
        fused: Tensor,
    ) -> Result<HeadOutputs, NtcError> {
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let ls = cfg.max_schema_tokens;
        let lu = cfg.max_utterance_tokens;
        let a = cfg.max_args;
        let e = cfg.max_enum_values;
        let n_tools = inputs.tools.len();

        let state_at = |idx: usize| -> &[f32] { &fused.data[idx * h..(idx + 1) * h] };
        let user_cls = &user_states.data[0..h];
        let global = state_at(n_tools * ls); // NO_TOOL

        // 5. Heads (mirrors CpuRefBackend::run exactly).
        let mut out: HashMap<String, Tensor> = HashMap::new();

        // action: concat(user_cls, global) -> mlp
        let mut cat = Vec::with_capacity(2 * h);
        cat.extend_from_slice(user_cls);
        cat.extend_from_slice(global);
        let action = self.mlp_head(&cat, "heads.action.dense", "heads.action.out")?;
        out.insert(
            "action.logits".into(),
            Tensor::from_vec(&[cfg.action_classes], action),
        );

        // tool: score each tool anchor + NO_TOOL
        let mut tool_logits = Vec::with_capacity(n_tools + 1);
        for (t, tool) in inputs.tools.iter().enumerate() {
            let s = state_at(t * ls + tool.tool_anchor);
            tool_logits.push(self.mlp_head(s, "heads.tool.dense", "heads.tool.out")?[0]);
        }
        tool_logits.push(self.mlp_head(global, "heads.tool.dense", "heads.tool.out")?[0]);
        out.insert(
            "tool.logits".into(),
            Tensor::from_vec(&[n_tools + 1], tool_logits),
        );

        // Per-arg heads.
        let mut presence = Tensor::from_vec(&[n_tools, a, 4], vec![f32::MIN; n_tools * a * 4]);
        let mut boolean = Tensor::from_vec(&[n_tools, a, 2], vec![f32::MIN; n_tools * a * 2]);
        let mut span_start = Tensor::from_vec(&[n_tools, a, lu], vec![f32::MIN; n_tools * a * lu]);
        let mut span_end = Tensor::from_vec(&[n_tools, a, lu], vec![f32::MIN; n_tools * a * lu]);
        let mut enum_logits = Tensor::from_vec(&[n_tools, a, e], vec![f32::MIN; n_tools * a * e]);
        let mut unit = Tensor::from_vec(&[n_tools, a, 6], vec![f32::MIN; n_tools * a * 6]);
        let mut magnitude = Tensor::zeros(&[n_tools, a, 1]);
        let mut relation = Tensor::from_vec(&[n_tools, a, 10], vec![f32::MIN; n_tools * a * 10]);
        let mut weekday = Tensor::from_vec(&[n_tools, a, 8], vec![f32::MIN; n_tools * a * 8]);
        let mut daypart = Tensor::from_vec(&[n_tools, a, 6], vec![f32::MIN; n_tools * a * 6]);
        let mut month = Tensor::from_vec(&[n_tools, a, 13], vec![f32::MIN; n_tools * a * 13]);

        for (t, tool) in inputs.tools.iter().enumerate() {
            for (k, &anchor) in tool.arg_anchors.iter().enumerate() {
                let arg_state = state_at(t * ls + anchor).to_vec();
                let base = t * a + k;

                presence.data[base * 4..base * 4 + 4].copy_from_slice(&self.mlp_head(
                    &arg_state,
                    "heads.presence.dense",
                    "heads.presence.out",
                )?);
                boolean.data[base * 2..base * 2 + 2]
                    .copy_from_slice(&self.linear_head(&arg_state, "heads.boolean.out")?);
                unit.data[base * 6..base * 6 + 6]
                    .copy_from_slice(&self.linear_head(&arg_state, "heads.numeric.unit")?);
                magnitude.data[base] = self.linear_head(&arg_state, "heads.numeric.magnitude")?[0];
                relation.data[base * 10..base * 10 + 10]
                    .copy_from_slice(&self.linear_head(&arg_state, "heads.datetime.relation")?);
                weekday.data[base * 8..base * 8 + 8]
                    .copy_from_slice(&self.linear_head(&arg_state, "heads.datetime.weekday")?);
                daypart.data[base * 6..base * 6 + 6]
                    .copy_from_slice(&self.linear_head(&arg_state, "heads.datetime.daypart")?);
                month.data[base * 13..base * 13 + 13]
                    .copy_from_slice(&self.linear_head(&arg_state, "heads.datetime.month")?);

                // Span heads over real utterance tokens.
                let n = inputs.utterance_len;
                let mut s_scores = vec![0.0f32; n];
                self.bilinear_scores(
                    &arg_state,
                    "heads.span.start.weight",
                    &user_states,
                    &mut s_scores,
                )?;
                span_start.data[base * lu..base * lu + n].copy_from_slice(&s_scores[..n]);
                let mut e_scores = vec![0.0f32; n];
                self.bilinear_scores(
                    &arg_state,
                    "heads.span.end.weight",
                    &user_states,
                    &mut e_scores,
                )?;
                span_end.data[base * lu..base * lu + n].copy_from_slice(&e_scores[..n]);

                // Enum head over this arg's enum-value anchors.
                let evs = &tool.enum_anchors[k];
                if !evs.is_empty() {
                    let mut targets = Tensor::zeros(&[evs.len(), h]);
                    for (j, &ea) in evs.iter().enumerate() {
                        targets.data[j * h..(j + 1) * h].copy_from_slice(state_at(t * ls + ea));
                    }
                    let mut scores = vec![0.0f32; evs.len()];
                    self.bilinear_scores(&arg_state, "heads.enum.weight", &targets, &mut scores)?;
                    enum_logits.data[base * e..base * e + evs.len()].copy_from_slice(&scores);
                }
            }
        }

        out.insert("presence.logits".into(), presence);
        out.insert("boolean.logits".into(), boolean);
        out.insert("span.start.logits".into(), span_start);
        out.insert("span.end.logits".into(), span_end);
        out.insert("enum.logits".into(), enum_logits);
        out.insert("numeric.unit.logits".into(), unit);
        out.insert("numeric.magnitude".into(), magnitude);
        out.insert("datetime.relation.logits".into(), relation);
        out.insert("datetime.weekday.logits".into(), weekday);
        out.insert("datetime.daypart.logits".into(), daypart);
        out.insert("datetime.month.logits".into(), month);

        // Head-codec v3, mirroring `CpuRefBackend`: these heads are small
        // projections over states already read back, so they run on the CPU
        // on this path too.
        if self.weights_have_v3 {
            out.insert(
                "delegate_reason.logits".into(),
                Tensor::from_vec(
                    &[4],
                    self.mlp_head(
                        &cat,
                        "heads.delegate_reason.dense",
                        "heads.delegate_reason.out",
                    )?,
                ),
            );
            out.insert(
                "no_call_reason.logits".into(),
                Tensor::from_vec(
                    &[5],
                    self.mlp_head(
                        &cat,
                        "heads.no_call_reason.dense",
                        "heads.no_call_reason.out",
                    )?,
                ),
            );

            let n_linked = inputs.linked_kinds.len();
            let width = ntc_model::inputs::MAX_LINKED + 1;
            let mut source = Tensor::from_vec(&[n_tools, a, 4], vec![f32::MIN; n_tools * a * 4]);
            let mut unresolved =
                Tensor::from_vec(&[n_tools, a, 5], vec![f32::MIN; n_tools * a * 5]);
            let mut entity =
                Tensor::from_vec(&[n_tools, a, width], vec![f32::MIN; n_tools * a * width]);

            let kind_emb = self.ht("context.linked_kind.weight")?.clone();
            let pos_emb = self.ht("context.linked_pos.weight")?.clone();
            let mut linked_states = Tensor::zeros(&[n_linked.max(1), h]);
            for (i, &kind) in inputs.linked_kinds.iter().enumerate() {
                let k = kind_emb.row(kind);
                let p = pos_emb.row(i);
                let row = &mut linked_states.data[i * h..(i + 1) * h];
                for j in 0..h {
                    row[j] = k[j] + p[j];
                }
            }
            let none_vec = self.ht("heads.entity.none.embedding")?.data.clone();

            for (t, tool) in inputs.tools.iter().enumerate() {
                for (k, &anchor) in tool.arg_anchors.iter().enumerate() {
                    let arg_state = state_at(t * ls + anchor).to_vec();
                    let base = t * a + k;
                    source.data[base * 4..base * 4 + 4]
                        .copy_from_slice(&self.linear_head(&arg_state, "heads.source.out")?);
                    unresolved.data[base * 5..base * 5 + 5].copy_from_slice(
                        &self.linear_head(&arg_state, "heads.unresolved_reason.out")?,
                    );
                    let mut scores = vec![0.0f32; n_linked];
                    if n_linked > 0 {
                        self.bilinear_scores(
                            &arg_state,
                            "heads.entity.proj.weight",
                            &linked_states,
                            &mut scores,
                        )?;
                    }
                    let row = &mut entity.data[base * width..base * width + width];
                    row[..n_linked].copy_from_slice(&scores);
                    let proj = ntc_model::ops::linear(
                        &Tensor::from_vec(&[1, h], arg_state.clone()),
                        self.ht("heads.entity.proj.weight")?,
                        None,
                    );
                    row[width - 1] = proj
                        .data
                        .iter()
                        .zip(&none_vec)
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                }
            }
            out.insert("source.logits".into(), source);
            out.insert("unresolved_reason.logits".into(), unresolved);
            out.insert("entity_ref.logits".into(), entity);
        }

        // Head-codec v4, mirroring `CpuRefBackend`. `decode` reads these
        // logits through `optional()`, so a backend that omits them does not
        // fail — it renders the raw span where the host declared a template.
        // That is a silent wrong answer, so parity here is deliberate.
        let k = self.cfg.filter_template_classes();
        if k > 0 && self.weights_have_v4 {
            let mut templates =
                Tensor::from_vec(&[n_tools, a, k], vec![f32::MIN; n_tools * a * k]);
            for (t, tool) in inputs.tools.iter().enumerate() {
                for (idx, &anchor) in tool.arg_anchors.iter().enumerate() {
                    let arg_state = state_at(t * ls + anchor).to_vec();
                    let base = t * a + idx;
                    templates.data[base * k..base * k + k].copy_from_slice(
                        &self.linear_head(&arg_state, "heads.filter_template.out")?,
                    );
                }
            }
            out.insert("filter_template.logits".into(), templates);
        }

        Ok(HeadOutputs { tensors: out })
    }
}

impl Backend for WgpuBackend {
    fn run(&mut self, inputs: &ModelInputs) -> Result<HeadOutputs, NtcError> {
        let (enc, user, fused) = self.record_forward(inputs)?;
        let (h, ls, lu) = (
            self.cfg.hidden,
            self.cfg.max_schema_tokens,
            self.cfg.max_utterance_tokens,
        );
        let s_len = inputs.tools.len() * ls + 1;
        let mut res = self
            .exec
            .read_back(enc, &[(&user, lu * h), (&fused, s_len * h)])?;
        let fused = Tensor::from_vec(&[s_len, h], res.pop().expect("fused states"));
        let user_states = Tensor::from_vec(&[lu, h], res.pop().expect("user states"));
        self.decode_heads(inputs, user_states, fused)
    }
}

impl ntc_model::AsyncBackend for WgpuBackend {
    /// Awaitable inference for wasm/WebGPU hosts (readback resolves via
    /// the browser event loop).
    async fn run_async(&mut self, inputs: &ModelInputs) -> Result<HeadOutputs, NtcError> {
        let (enc, user, fused) = self.record_forward(inputs)?;
        let (h, ls, lu) = (
            self.cfg.hidden,
            self.cfg.max_schema_tokens,
            self.cfg.max_utterance_tokens,
        );
        let s_len = inputs.tools.len() * ls + 1;
        let mut res = self
            .exec
            .read_back_async(enc, &[(&user, lu * h), (&fused, s_len * h)])
            .await?;
        let fused = Tensor::from_vec(&[s_len, h], res.pop().expect("fused states"));
        let user_states = Tensor::from_vec(&[lu, h], res.pop().expect("user states"));
        self.decode_heads(inputs, user_states, fused)
    }
}
