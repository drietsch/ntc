//! Shared test/fixture helpers (feature `test-support`): a tiny architecture
//! config, a deterministic word-level test tokenizer, and seeded random
//! weights. Used by unit tests across crates and by `ntc fixture-gen`.
//!
//! The test tokenizer is NOT the product tokenizer — it exists so runtime
//! plumbing is testable with a ~60-word vocab and byte offsets that behave
//! like the real (SentencePiece-style) tokenizer's.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::config::{Calibration, NtcArchConfig};
use crate::tensor::Tensor;
use crate::weights::{tensor_specs, ModelWeights};

/// Tiny dims: fast on CPU, exercises every code path.
pub fn tiny_config() -> NtcArchConfig {
    NtcArchConfig {
        hidden: 32,
        heads: 4,
        ffn: 64,
        vocab: 64,
        max_positions: 128,
        encoder_layers: 2,
        schema_layers: 1,
        fusion_blocks: 1,
        max_tools: 4,
        max_args: 4,
        max_enum_values: 4,
        max_utterance_tokens: 24,
        max_schema_tokens: 64,
        layer_norm_eps: 1e-5,
        action_classes: 3,
        calibration: Calibration::default(),
    }
}

/// Word-level tokenizer with whitespace pre-tokenization, `<s>`/`</s>`
/// wrapping, and an `<unk>` fallback. Ids fit in [`tiny_config`]'s vocab.
pub fn test_tokenizer_json() -> String {
    let words = [
        "TOOL",
        "DESC",
        "ARG",
        "INFO",
        "TYPE",
        "REQUIRED",
        "SEMANTIC",
        "ENUM",
        "TEXT",
        "INTEGER",
        "FLOAT",
        "BOOLEAN",
        "DATE",
        "DATETIME",
        "DURATION",
        "PERSON",
        "LOCATION",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "create",
        "a",
        "calendar",
        "event",
        "send",
        "an",
        "email",
        "title",
        "start",
        "priority",
        "low",
        "normal",
        "high",
        "recipient",
        "subject",
        "make",
        "appointment",
        "tomorrow",
        "afternoon",
        "dentist",
        "one",
        "hour",
        "the",
        "to",
    ];
    let mut vocab = serde_json::Map::new();
    vocab.insert("<s>".into(), 0.into());
    vocab.insert("</s>".into(), 1.into());
    vocab.insert("<unk>".into(), 2.into());
    for (i, w) in words.iter().enumerate() {
        vocab.insert((*w).into(), (3 + i).into());
    }

    let added = |id: u32, content: &str| {
        serde_json::json!({
            "id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true
        })
    };

    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [added(0, "<s>"), added(1, "</s>"), added(2, "<unk>")],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "</s>", "type_id": 1}}
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}
            }
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
    .to_string()
}

/// Seeded uniform weights in ±0.5/√hidden (keeps activations tame through
/// the post-LN stack). Deterministic per (cfg, seed).
pub fn random_weights(cfg: &NtcArchConfig, seed: u64) -> ModelWeights {
    random_weights_for(cfg, seed, false)
}

/// The same weights plus the optional v3 head tensors.
///
/// Kept separate from [`random_weights`] deliberately: the v3 specs are
/// appended *after* the v2 ones and draw from the same RNG stream, so every
/// v2 tensor is bit-identical between the two. Unit tests that pin golden
/// logits keep using the v2 set; the exported fixture uses this one, so it
/// carries the full head set the shipping model actually has and can be
/// cross-checked name-for-name against the Python exporter.
pub fn random_weights_with_v3(cfg: &NtcArchConfig, seed: u64) -> ModelWeights {
    random_weights_for(cfg, seed, true)
}

fn random_weights_for(cfg: &NtcArchConfig, seed: u64, with_v3: bool) -> ModelWeights {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let scale = 0.5 / (cfg.hidden as f32).sqrt();
    let mut map = std::collections::HashMap::new();
    let mut specs = tensor_specs(cfg);
    if with_v3 {
        specs.extend(crate::weights::v3_head_specs(cfg));
    }
    for (name, shape) in specs {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = if name.ends_with("norm.weight") {
            vec![1.0; n]
        } else if name.ends_with("norm.bias") || name.ends_with(".bias") {
            vec![0.0; n]
        } else {
            (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
        };
        map.insert(name, Tensor::from_vec(&shape, data));
    }
    ModelWeights::from_map(map)
}
