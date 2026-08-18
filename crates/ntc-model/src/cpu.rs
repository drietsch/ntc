//! CPU reference backend: the normative forward pass (naive f32).
//!
//! Every GPU backend must reproduce these head outputs within tolerance
//! (`fixtures/tolerances.toml`) and with 100% decision parity.

#![allow(clippy::needless_range_loop)] // reference-code clarity

use std::collections::HashMap;

use ntc_core::NtcError;

use crate::backend::{Backend, HeadOutputs};
use crate::config::NtcArchConfig;
use crate::inputs::{ModelInputs, ToolInput};
use crate::ops;
use crate::tensor::Tensor;
use crate::weights::ModelWeights;

pub struct CpuRefBackend {
    cfg: NtcArchConfig,
    weights: ModelWeights,
}

impl CpuRefBackend {
    pub fn new(cfg: NtcArchConfig, weights: ModelWeights) -> Self {
        Self { cfg, weights }
    }

    pub fn config(&self) -> &NtcArchConfig {
        &self.cfg
    }

    fn w(&self, name: &str) -> Result<&Tensor, NtcError> {
        self.weights.get(name)
    }

    fn wb(&self, prefix: &str) -> Result<(&Tensor, &Tensor), NtcError> {
        Ok((
            self.w(&format!("{prefix}.weight"))?,
            self.w(&format!("{prefix}.bias"))?,
        ))
    }

    /// One post-LN transformer layer (self-attention) in place.
    fn transformer_layer(
        &self,
        prefix: &str,
        states: &mut Tensor,
        mask: &[bool],
    ) -> Result<(), NtcError> {
        let eps = self.cfg.layer_norm_eps;
        let attn = ops::attention(
            states,
            states,
            mask,
            self.wb(&format!("{prefix}.attn.q"))?,
            self.wb(&format!("{prefix}.attn.k"))?,
            self.wb(&format!("{prefix}.attn.v"))?,
            self.wb(&format!("{prefix}.attn.o"))?,
            self.cfg.heads,
        );
        ops::add_inplace(states, &attn);
        let (nw, nb) = self.wb(&format!("{prefix}.attn.norm"))?;
        *states = ops::layer_norm(states, nw, nb, eps);

        let mut up = ops::linear(
            states,
            self.w(&format!("{prefix}.ffn.up.weight"))?,
            Some(self.w(&format!("{prefix}.ffn.up.bias"))?),
        );
        ops::gelu(&mut up);
        let down = ops::linear(
            &up,
            self.w(&format!("{prefix}.ffn.down.weight"))?,
            Some(self.w(&format!("{prefix}.ffn.down.bias"))?),
        );
        ops::add_inplace(states, &down);
        let (nw, nb) = self.wb(&format!("{prefix}.ffn.norm"))?;
        *states = ops::layer_norm(states, nw, nb, eps);
        Ok(())
    }

    fn embed_utterance(&self, ids: &[u32]) -> Result<Tensor, NtcError> {
        let h = self.cfg.hidden;
        let word = self.w("embeddings.word.weight")?;
        let pos = self.w("embeddings.position.weight")?;
        let mut x = Tensor::zeros(&[ids.len(), h]);
        for (i, &id) in ids.iter().enumerate() {
            let wrow = word.row(id as usize);
            let prow = pos.row(i);
            let o = &mut x.data[i * h..(i + 1) * h];
            for j in 0..h {
                o[j] = wrow[j] + prow[j];
            }
        }
        let (nw, nb) = self.wb("embeddings.norm")?;
        Ok(ops::layer_norm(&x, nw, nb, self.cfg.layer_norm_eps))
    }

    fn encode_utterance(&self, inputs: &ModelInputs) -> Result<Tensor, NtcError> {
        let mut states = self.embed_utterance(&inputs.utterance_ids)?;
        for i in 0..self.cfg.encoder_layers {
            self.transformer_layer(
                &format!("encoder.layer.{i}"),
                &mut states,
                &inputs.utterance_mask,
            )?;
        }
        Ok(states)
    }

    fn encode_tool(&self, tool: &ToolInput, tool_index: usize) -> Result<Tensor, NtcError> {
        let h = self.cfg.hidden;
        let word = self.w("embeddings.word.weight")?;
        let pos = self.w("embeddings.position.weight")?;
        let kind_emb = self.w("schema.embeddings.segment_kind.weight")?;
        let tool_emb = self.w("schema.embeddings.tool_index.weight")?;
        let trow = tool_emb.row(tool_index);

        let ls = tool.ids.len();
        let mut x = Tensor::zeros(&[ls, h]);
        for i in 0..ls {
            let wrow = word.row(tool.ids[i] as usize);
            let prow = pos.row(i);
            let krow = kind_emb.row(tool.kinds[i] as usize);
            let o = &mut x.data[i * h..(i + 1) * h];
            for j in 0..h {
                o[j] = wrow[j] + prow[j] + krow[j] + trow[j];
            }
        }
        let (nw, nb) = self.wb("schema.embeddings.norm")?;
        let mut states = ops::layer_norm(&x, nw, nb, self.cfg.layer_norm_eps);
        for i in 0..self.cfg.schema_layers {
            self.transformer_layer(&format!("schema.layer.{i}"), &mut states, &tool.mask)?;
        }
        Ok(states)
    }

    /// Fusion over the packed schema sequence (+ NO_TOOL slot at the end).
    fn fuse(
        &self,
        tool_states: &[Tensor],
        packed_mask: &[bool],
        user_states: &Tensor,
        user_mask: &[bool],
    ) -> Result<Tensor, NtcError> {
        let h = self.cfg.hidden;
        let ls = self.cfg.max_schema_tokens;
        let s = tool_states.len() * ls + 1;
        let mut packed = Tensor::zeros(&[s, h]);
        for (t, st) in tool_states.iter().enumerate() {
            packed.data[t * ls * h..(t + 1) * ls * h].copy_from_slice(&st.data);
        }
        let no_tool = self.w("fusion.no_tool.embedding")?;
        packed.data[(s - 1) * h..].copy_from_slice(&no_tool.data);

        let eps = self.cfg.layer_norm_eps;
        for i in 0..self.cfg.fusion_blocks {
            let p = format!("fusion.block.{i}");
            let sa = ops::attention(
                &packed,
                &packed,
                packed_mask,
                self.wb(&format!("{p}.self.q"))?,
                self.wb(&format!("{p}.self.k"))?,
                self.wb(&format!("{p}.self.v"))?,
                self.wb(&format!("{p}.self.o"))?,
                self.cfg.heads,
            );
            ops::add_inplace(&mut packed, &sa);
            let (nw, nb) = self.wb(&format!("{p}.self.norm"))?;
            packed = ops::layer_norm(&packed, nw, nb, eps);

            let ca = ops::attention(
                &packed,
                user_states,
                user_mask,
                self.wb(&format!("{p}.cross.q"))?,
                self.wb(&format!("{p}.cross.k"))?,
                self.wb(&format!("{p}.cross.v"))?,
                self.wb(&format!("{p}.cross.o"))?,
                self.cfg.heads,
            );
            ops::add_inplace(&mut packed, &ca);
            let (nw, nb) = self.wb(&format!("{p}.cross.norm"))?;
            packed = ops::layer_norm(&packed, nw, nb, eps);

            let mut up = ops::linear(
                &packed,
                self.w(&format!("{p}.ffn.up.weight"))?,
                Some(self.w(&format!("{p}.ffn.up.bias"))?),
            );
            ops::gelu(&mut up);
            let down = ops::linear(
                &up,
                self.w(&format!("{p}.ffn.down.weight"))?,
                Some(self.w(&format!("{p}.ffn.down.bias"))?),
            );
            ops::add_inplace(&mut packed, &down);
            let (nw, nb) = self.wb(&format!("{p}.ffn.norm"))?;
            packed = ops::layer_norm(&packed, nw, nb, eps);
        }
        Ok(packed)
    }

    /// `dense -> gelu -> out` MLP head on a single state row; returns logits.
    fn mlp_head(&self, state: &[f32], dense: &str, out: &str) -> Result<Vec<f32>, NtcError> {
        let x = Tensor::from_vec(&[1, state.len()], state.to_vec());
        let (dw, db) = self.wb(dense)?;
        let mut hidden = ops::linear(&x, dw, Some(db));
        ops::gelu(&mut hidden);
        let (ow, ob) = self.wb(out)?;
        Ok(ops::linear(&hidden, ow, ob.into()).data)
    }

    /// Plain linear head on a single state row.
    fn linear_head(&self, state: &[f32], prefix: &str) -> Result<Vec<f32>, NtcError> {
        let x = Tensor::from_vec(&[1, state.len()], state.to_vec());
        let (w, b) = self.wb(prefix)?;
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
        let w = self.w(w_name)?;
        let x = Tensor::from_vec(&[1, state.len()], state.to_vec());
        let proj = ops::linear(&x, w, None);
        let h = state.len();
        for (j, o) in out.iter_mut().enumerate() {
            let row = &targets.data[j * h..(j + 1) * h];
            let mut dot = 0.0;
            for d in 0..h {
                dot += proj.data[d] * row[d];
            }
            *o = dot;
        }
        Ok(())
    }
}

impl Backend for CpuRefBackend {
    fn run(&mut self, inputs: &ModelInputs) -> Result<HeadOutputs, NtcError> {
        let cfg = &self.cfg;
        let h = cfg.hidden;
        let ls = cfg.max_schema_tokens;
        let lu = cfg.max_utterance_tokens;
        let a = cfg.max_args;
        let e = cfg.max_enum_values;
        let n_tools = inputs.tools.len();

        // 1. Encoders.
        let user_states = self.encode_utterance(inputs)?;
        let tool_states: Vec<Tensor> = inputs
            .tools
            .iter()
            .enumerate()
            .map(|(t, tool)| self.encode_tool(tool, t))
            .collect::<Result<_, _>>()?;

        // 2. Fusion.
        let mut packed_mask = Vec::with_capacity(n_tools * ls + 1);
        for tool in &inputs.tools {
            packed_mask.extend_from_slice(&tool.mask);
        }
        packed_mask.push(true); // NO_TOOL slot
        let fused = self.fuse(
            &tool_states,
            &packed_mask,
            &user_states,
            &inputs.utterance_mask,
        )?;

        let state_at = |idx: usize| -> &[f32] { &fused.data[idx * h..(idx + 1) * h] };
        let user_cls = &user_states.data[0..h];
        let global = state_at(n_tools * ls); // NO_TOOL

        // 3. Heads.
        let mut out: HashMap<String, Tensor> = HashMap::new();

        // action: concat(user_cls, global) -> mlp
        let mut cat = Vec::with_capacity(2 * h);
        cat.extend_from_slice(user_cls);
        cat.extend_from_slice(global);
        let action = self.mlp_head(&cat, "heads.action.dense", "heads.action.out")?;
        out.insert("action.logits".into(), Tensor::from_vec(&[3], action));

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

        Ok(HeadOutputs { tensors: out })
    }
}
