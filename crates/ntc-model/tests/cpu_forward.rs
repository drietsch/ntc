//! End-to-end CPU reference forward pass on the tiny fixture model:
//! anchor discovery, shape contract, determinism.

use ntc_core::schema::{compile_schema, RawToolSchema};
use ntc_core::tokenizer::NtcTokenizer;
use ntc_model::test_support::{random_weights, test_tokenizer_json, tiny_config};
use ntc_model::{Backend, CpuRefBackend, ModelInputs};

fn tools() -> Vec<ntc_core::CanonicalTool> {
    let calendar: RawToolSchema = serde_json::from_value(serde_json::json!({
        "name": "calendar.create",
        "description": "create a calendar event",
        "parameters": {
            "title": {"type": "string", "required": true},
            "start": {"type": "string", "format": "date-time", "required": true},
            "priority": {"type": "string", "enum": ["low", "normal", "high"]}
        }
    }))
    .unwrap();
    let email: RawToolSchema = serde_json::from_value(serde_json::json!({
        "name": "email.send",
        "description": "send an email",
        "parameters": {
            "recipient": {"type": "string", "required": true},
            "subject": {"type": "string"}
        }
    }))
    .unwrap();
    vec![
        compile_schema(&calendar).unwrap(),
        compile_schema(&email).unwrap(),
    ]
}

#[test]
fn pack_and_forward() {
    let cfg = tiny_config();
    let tokenizer = NtcTokenizer::from_bytes(test_tokenizer_json().as_bytes()).unwrap();
    let tools = tools();
    let refs: Vec<&_> = tools.iter().collect();

    let utterance = tokenizer
        .encode_utterance("make a dentist appointment tomorrow afternoon")
        .unwrap();
    // <s> + 6 words + </s>
    assert_eq!(utterance.ids.len(), 8);
    assert_eq!(utterance.ids[0], 0);

    let inputs = ModelInputs::pack(&cfg, &tokenizer, &utterance, &refs).unwrap();
    assert_eq!(inputs.tools.len(), 2);
    assert_eq!(inputs.utterance_len, 8);

    // Anchor sanity: the calendar tool's arg anchors must point at tokens
    // whose segment kind is ARG_NAME (kind 3).
    let cal = &inputs.tools[0];
    assert_eq!(cal.arg_anchors.len(), 3);
    for &a in &cal.arg_anchors {
        assert_eq!(cal.kinds[a], 3, "arg anchor must sit on an ARG line");
        assert!(cal.mask[a]);
    }
    // priority has 3 enum anchors on ENUM lines (kind 8).
    assert_eq!(cal.enum_anchors[2].len(), 3);
    for &e in &cal.enum_anchors[2] {
        assert_eq!(cal.kinds[e], 8);
    }

    let weights = random_weights(&cfg, 7);
    let mut backend = CpuRefBackend::new(cfg.clone(), weights);
    let out = backend.run(&inputs).unwrap();

    // Shape contract (head codec).
    assert_eq!(out.get("action.logits").unwrap().shape, vec![3]);
    assert_eq!(out.get("tool.logits").unwrap().shape, vec![3]); // 2 tools + NO_TOOL
    assert_eq!(
        out.get("presence.logits").unwrap().shape,
        vec![2, cfg.max_args, 4]
    );
    assert_eq!(
        out.get("span.start.logits").unwrap().shape,
        vec![2, cfg.max_args, cfg.max_utterance_tokens]
    );
    assert_eq!(
        out.get("enum.logits").unwrap().shape,
        vec![2, cfg.max_args, cfg.max_enum_values]
    );

    // Valid-region logits are finite; padded arg slots stay at f32::MIN.
    let presence = out.get("presence.logits").unwrap();
    for k in 0..3 {
        for c in 0..4 {
            let v = presence.data[(k * 4) + c];
            assert!(v.is_finite() && v > f32::MIN / 2.0);
        }
    }
    let pad_slot = presence.data[(3 * 4)..(3 * 4) + 4].to_vec(); // arg 3 undeclared
    assert!(pad_slot.iter().all(|&v| v == f32::MIN));

    // Enum logits: only 3 declared values get scores.
    let enums = out.get("enum.logits").unwrap();
    let e = cfg.max_enum_values;
    let prio = &enums.data[(2 * e)..(2 * e) + e]; // tool 0, arg 2
    assert!(prio[..3]
        .iter()
        .all(|v| v.is_finite() && *v > f32::MIN / 2.0));
    assert_eq!(prio[3], f32::MIN);

    // Determinism: same inputs → bit-identical outputs.
    let out2 = backend.run(&inputs).unwrap();
    for (name, t) in &out.tensors {
        assert_eq!(
            &out2.tensors[name].data, &t.data,
            "{name} not deterministic"
        );
    }
}

#[test]
fn span_text_resolution_round_trip() {
    let tokenizer = NtcTokenizer::from_bytes(test_tokenizer_json().as_bytes()).unwrap();
    let text = "make a dentist appointment tomorrow afternoon";
    let seq = tokenizer.encode_utterance(text).unwrap();
    // tokens: <s> make a dentist appointment tomorrow afternoon </s>
    let resolved = seq.span_text(text, 3, 5).unwrap();
    assert_eq!(resolved, "dentist appointment");
    // Span touching the trailing special token skips it.
    assert_eq!(seq.span_text(text, 5, 8).unwrap(), "tomorrow afternoon");
    assert!(seq.span_text(text, 5, 5).is_none());
}
