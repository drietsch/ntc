//! Model weights: the canonical tensor-name contract and `.ntc` loading.
//!
//! Naming convention (normative; the Python exporter renames PyTorch
//! parameters to exactly these names, transposing linear weights to
//! **[in, out]** so kernels compute `y = x·W + b`):
//!
//! ```text
//! embeddings.word.weight                 [vocab, H]
//! embeddings.position.weight             [max_positions, H]
//! embeddings.norm.{weight,bias}          [H]
//! encoder.layer.{i}.attn.{q,k,v,o}.{weight,bias}
//! encoder.layer.{i}.attn.norm.{weight,bias}
//! encoder.layer.{i}.ffn.up.{weight,bias}      [H, F] / [F]
//! encoder.layer.{i}.ffn.down.{weight,bias}    [F, H] / [H]
//! encoder.layer.{i}.ffn.norm.{weight,bias}
//! schema.embeddings.segment_kind.weight  [11, H]
//! schema.embeddings.tool_index.weight    [max_tools+1, H]
//! schema.embeddings.norm.{weight,bias}
//! schema.layer.{i}.*                     (same shape family as encoder.layer)
//! fusion.no_tool.embedding               [H]
//! fusion.block.{i}.self.{q,k,v,o}.{weight,bias}
//! fusion.block.{i}.self.norm.{weight,bias}
//! fusion.block.{i}.cross.{q,k,v,o}.{weight,bias}
//! fusion.block.{i}.cross.norm.{weight,bias}
//! fusion.block.{i}.ffn.{up,down}.{weight,bias}
//! fusion.block.{i}.ffn.norm.{weight,bias}
//! heads.action.dense.{weight,bias}       [2H, H] / [H]
//! heads.action.out.{weight,bias}         [H, 3]  / [3]
//! heads.tool.dense.{weight,bias}         [H, H]  / [H]
//! heads.tool.out.{weight,bias}           [H, 1]  / [1]
//! heads.presence.dense.{weight,bias}     [H, H]  / [H]
//! heads.presence.out.{weight,bias}       [H, 4]  / [4]
//! heads.boolean.out.{weight,bias}        [H, 2]  / [2]
//! heads.span.start.weight                [H, H]   (bilinear, no bias)
//! heads.span.end.weight                  [H, H]
//! heads.enum.weight                      [H, H]
//! heads.numeric.unit.{weight,bias}       [H, 6]  / [6]
//! heads.numeric.magnitude.{weight,bias}  [H, 1]  / [1]
//! heads.datetime.relation.{weight,bias}  [H, 10] / [10]
//! heads.datetime.weekday.{weight,bias}   [H, 8]  / [8]
//! heads.datetime.daypart.{weight,bias}   [H, 6]  / [6]
//! heads.datetime.month.{weight,bias}     [H, 13] / [13]
//! ```
//!
//! Exporter normalizations (documented for the training side):
//! - XLM-R position rows are de-offset (runtime indexes 0..L directly),
//! - the single token-type embedding row is folded into position embeddings.

use std::collections::HashMap;

use ntc_core::NtcError;
use ntc_format::NtcFile;

use crate::config::NtcArchConfig;
use crate::tensor::Tensor;

/// Number of segment-kind embedding rows (see [`crate::inputs::SegmentKind`]).
pub const SEGMENT_KINDS: usize = 11;

#[derive(Debug)]
pub struct ModelWeights {
    tensors: HashMap<String, Tensor>,
}

impl ModelWeights {
    /// Decode all tensors from a parsed `.ntc` file and check the tensor set
    /// against the architecture config (presence + shapes of the load-bearing
    /// tensors; per-layer families checked for every declared layer).
    pub fn from_ntc(file: &NtcFile<'_>, cfg: &NtcArchConfig) -> Result<Self, NtcError> {
        let mut tensors = HashMap::with_capacity(file.records.len());
        for view in file.tensors() {
            tensors.insert(view.record.name.clone(), Tensor::from_view(&view)?);
        }
        let w = Self { tensors };
        w.check(cfg)?;
        Ok(w)
    }

    pub fn from_map(tensors: HashMap<String, Tensor>) -> Self {
        Self { tensors }
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, NtcError> {
        self.tensors
            .get(name)
            .ok_or_else(|| NtcError::Format(format!("missing tensor `{name}`")))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Is a tensor present? Used to detect optional (head-codec v3) heads.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// True when this model carries the head-codec v3 heads (reasons,
    /// argument source, entity reference). Models exported before v3 do not,
    /// and the backends simply skip those outputs.
    pub fn has_v3_heads(&self) -> bool {
        self.has("heads.delegate_reason.out.weight") && self.has("heads.entity.proj.weight")
    }

    /// True when this model carries the head-codec v4 filter-template head.
    /// Its width comes from the declared template table, so a model that
    /// declares no templates has no head and no tensors.
    pub fn has_filter_template_head(&self) -> bool {
        self.has("heads.filter_template.out.weight")
    }

    /// Check presence + shape of every tensor [`tensor_specs`] declares.
    pub fn check(&self, cfg: &NtcArchConfig) -> Result<(), NtcError> {
        if self.has_filter_template_head() != (cfg.filter_template_classes() > 0) {
            return Err(NtcError::Format(format!(
                "model declares {} filter templates but {} a filter-template head",
                cfg.filter_templates.len(),
                if self.has_filter_template_head() { "carries" } else { "lacks" },
            )));
        }
        if self.has_filter_template_head() {
            for (name, shape) in v4_head_specs(cfg) {
                let t = self.get(&name)?;
                if t.shape != shape {
                    return Err(NtcError::Format(format!(
                        "tensor `{name}`: shape {:?}, expected {shape:?}",
                        t.shape
                    )));
                }
            }
        }
        if self.has_v3_heads() {
            for (name, shape) in v3_head_specs(cfg) {
                let t = self.get(&name)?;
                if t.shape != shape {
                    return Err(NtcError::Format(format!(
                        "tensor `{name}`: shape {:?}, expected {shape:?}",
                        t.shape
                    )));
                }
            }
        }
        for (name, shape) in tensor_specs(cfg) {
            let t = self.get(&name)?;
            if t.shape != shape {
                return Err(NtcError::Format(format!(
                    "tensor `{name}`: shape {:?}, expected {shape:?}",
                    t.shape
                )));
            }
        }
        Ok(())
    }
}

/// Head-codec v3 tensors: present only on models exported with the reason,
/// source and entity-reference heads.
pub fn v3_head_specs(cfg: &NtcArchConfig) -> Vec<(String, Vec<usize>)> {
    let h = cfg.hidden;
    let mut specs = vec![
        (
            "heads.delegate_reason.dense.weight".to_string(),
            vec![2 * h, h],
        ),
        ("heads.delegate_reason.dense.bias".to_string(), vec![h]),
        ("heads.delegate_reason.out.weight".to_string(), vec![h, 4]),
        ("heads.delegate_reason.out.bias".to_string(), vec![4]),
        (
            "heads.no_call_reason.dense.weight".to_string(),
            vec![2 * h, h],
        ),
        ("heads.no_call_reason.dense.bias".to_string(), vec![h]),
        ("heads.no_call_reason.out.weight".to_string(), vec![h, 5]),
        ("heads.no_call_reason.out.bias".to_string(), vec![5]),
        ("heads.source.out.weight".to_string(), vec![h, 4]),
        ("heads.source.out.bias".to_string(), vec![4]),
        ("heads.unresolved_reason.out.weight".to_string(), vec![h, 5]),
        ("heads.unresolved_reason.out.bias".to_string(), vec![5]),
        ("heads.entity.proj.weight".to_string(), vec![h, h]),
        ("heads.entity.none.embedding".to_string(), vec![h]),
    ];
    specs.push((
        "context.linked_kind.weight".to_string(),
        vec![crate::inputs::LINKED_KINDS.len() + 1, h],
    ));
    specs.push((
        "context.linked_pos.weight".to_string(),
        vec![crate::inputs::MAX_LINKED + 1, h],
    ));
    specs
}

/// Head-codec v4 tensors: the filter-template head, present only on models
/// that declare a template table.
pub fn v4_head_specs(cfg: &NtcArchConfig) -> Vec<(String, Vec<usize>)> {
    let k = cfg.filter_template_classes();
    if k == 0 {
        return vec![];
    }
    vec![
        (
            "heads.filter_template.out.weight".to_string(),
            vec![cfg.hidden, k],
        ),
        ("heads.filter_template.out.bias".to_string(), vec![k]),
    ]
}

/// The complete tensor manifest for an architecture config: every canonical
/// tensor name with its expected shape. Single source for weight checking,
/// fixture generation, and GPU buffer setup.
pub fn tensor_specs(cfg: &NtcArchConfig) -> Vec<(String, Vec<usize>)> {
    let h = cfg.hidden;
    let f = cfg.ffn;
    let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| specs.push((name, shape));

    push("embeddings.word.weight".into(), vec![cfg.vocab, h]);
    push(
        "embeddings.position.weight".into(),
        vec![cfg.max_positions, h],
    );
    push("embeddings.norm.weight".into(), vec![h]);
    push("embeddings.norm.bias".into(), vec![h]);

    let layer = |specs: &mut Vec<(String, Vec<usize>)>, prefix: String| {
        for p in ["q", "k", "v", "o"] {
            specs.push((format!("{prefix}.attn.{p}.weight"), vec![h, h]));
            specs.push((format!("{prefix}.attn.{p}.bias"), vec![h]));
        }
        specs.push((format!("{prefix}.attn.norm.weight"), vec![h]));
        specs.push((format!("{prefix}.attn.norm.bias"), vec![h]));
        specs.push((format!("{prefix}.ffn.up.weight"), vec![h, f]));
        specs.push((format!("{prefix}.ffn.up.bias"), vec![f]));
        specs.push((format!("{prefix}.ffn.down.weight"), vec![f, h]));
        specs.push((format!("{prefix}.ffn.down.bias"), vec![h]));
        specs.push((format!("{prefix}.ffn.norm.weight"), vec![h]));
        specs.push((format!("{prefix}.ffn.norm.bias"), vec![h]));
    };
    for i in 0..cfg.encoder_layers {
        layer(&mut specs, format!("encoder.layer.{i}"));
    }

    specs.push((
        "schema.embeddings.segment_kind.weight".into(),
        vec![SEGMENT_KINDS, h],
    ));
    specs.push((
        "schema.embeddings.tool_index.weight".into(),
        vec![cfg.max_tools + 1, h],
    ));
    specs.push(("schema.embeddings.norm.weight".into(), vec![h]));
    specs.push(("schema.embeddings.norm.bias".into(), vec![h]));
    for i in 0..cfg.schema_layers {
        layer(&mut specs, format!("schema.layer.{i}"));
    }

    specs.push(("fusion.no_tool.embedding".into(), vec![h]));
    for i in 0..cfg.fusion_blocks {
        for part in ["self", "cross"] {
            for p in ["q", "k", "v", "o"] {
                specs.push((format!("fusion.block.{i}.{part}.{p}.weight"), vec![h, h]));
                specs.push((format!("fusion.block.{i}.{part}.{p}.bias"), vec![h]));
            }
            specs.push((format!("fusion.block.{i}.{part}.norm.weight"), vec![h]));
            specs.push((format!("fusion.block.{i}.{part}.norm.bias"), vec![h]));
        }
        specs.push((format!("fusion.block.{i}.ffn.up.weight"), vec![h, f]));
        specs.push((format!("fusion.block.{i}.ffn.up.bias"), vec![f]));
        specs.push((format!("fusion.block.{i}.ffn.down.weight"), vec![f, h]));
        specs.push((format!("fusion.block.{i}.ffn.down.bias"), vec![h]));
        specs.push((format!("fusion.block.{i}.ffn.norm.weight"), vec![h]));
        specs.push((format!("fusion.block.{i}.ffn.norm.bias"), vec![h]));
    }

    // Heads (shapes per contracts/heads/v1/head-spec.json).
    for (dense, out, classes) in [
        ("heads.action.dense", "heads.action.out", cfg.action_classes),
        ("heads.tool.dense", "heads.tool.out", 1),
        ("heads.presence.dense", "heads.presence.out", 4),
    ] {
        let in_dim = if dense == "heads.action.dense" {
            2 * h
        } else {
            h
        };
        specs.push((format!("{dense}.weight"), vec![in_dim, h]));
        specs.push((format!("{dense}.bias"), vec![h]));
        specs.push((format!("{out}.weight"), vec![h, classes]));
        specs.push((format!("{out}.bias"), vec![classes]));
    }
    for name in [
        "heads.span.start.weight",
        "heads.span.end.weight",
        "heads.enum.weight",
    ] {
        specs.push((name.to_string(), vec![h, h]));
    }
    for (name, classes) in [
        ("heads.boolean.out", 2usize),
        ("heads.numeric.unit", 6),
        ("heads.numeric.magnitude", 1),
        ("heads.datetime.relation", 10),
        ("heads.datetime.weekday", 8),
        ("heads.datetime.daypart", 6),
        ("heads.datetime.month", 13),
    ] {
        specs.push((format!("{name}.weight"), vec![h, classes]));
        specs.push((format!("{name}.bias"), vec![classes]));
    }
    specs
}
