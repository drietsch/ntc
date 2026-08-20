//! Head codec v4: a value the utterance does not contain, built from a
//! template the model chose and the span it marked.
//!
//! The heads are stubbed rather than run, so what is under test is the decode
//! contract — which class index means which template, that masking keeps an
//! argument out of templates meant for a different annotation, and that an
//! unfillable slot refuses instead of guessing — not whether some checkpoint
//! happens to predict them.

use ntc_core::ir::SemanticValue;
use ntc_core::schema::{compile_schema, CanonicalTool, RawToolSchema};
use ntc_core::tokenizer::NtcTokenizer;
use ntc_model::config::FilterTemplate;
use ntc_model::test_support::{test_tokenizer_json, tiny_config};
use ntc_model::{HeadOutputs, ModelInputs, Tensor};
use ntc_runtime::decode::Decoder;

fn templates() -> Vec<FilterTemplate> {
    vec![
        FilterTemplate {
            id: "FIELD_IS_NULL".into(),
            semantic: "FILTER.PQL".into(),
            pattern: "{field} IS NULL".into(),
            values: vec![],
        },
        FilterTemplate {
            id: "FIELD_LESS_THAN".into(),
            semantic: "FILTER.PQL".into(),
            pattern: "{field} < {number}".into(),
            values: vec![],
        },
        FilterTemplate {
            id: "UNPUBLISHED_PAGES".into(),
            semantic: "FILTER.PQL".into(),
            pattern: r#"type = "page" AND published = false"#.into(),
            values: vec![],
        },
        FilterTemplate {
            id: "SOMETHING_ELSE".into(),
            // A template for a different annotation entirely: it must never be
            // reachable from a FILTER.PQL argument, however high its logit.
            semantic: "SORT.ORDER".into(),
            pattern: "id DESC".into(),
            values: vec![],
        },
    ]
}

fn tool() -> CanonicalTool {
    let raw: RawToolSchema = serde_json::from_value(serde_json::json!({
        "name": "search_data_objects",
        "description": "search data objects",
        "parameters": {
            "className": {"type": "string"},
            "pqlFilter": {"type": "string", "semantic": "FILTER.PQL"},
            "parentId": {"type": "integer", "default": 1}
        }
    }))
    .unwrap();
    compile_schema(&raw).unwrap()
}

/// Head outputs with `pqlFilter` (arg 1) pointed at `template_class` and its
/// span covering `[span_start, span_end]` token indices, inclusive.
fn outputs(
    cfg: &ntc_model::NtcArchConfig,
    n_classes: usize,
    template_logits: &[(usize, f32)],
    span: Option<(usize, usize)>,
    source_class: usize,
) -> HeadOutputs {
    let (t, a, lu) = (1usize, cfg.max_args, cfg.max_utterance_tokens);
    let arg = 1usize; // pqlFilter

    let mut ft = Tensor::from_vec(&[t, a, n_classes], vec![0.0; t * a * n_classes]);
    for &(class, logit) in template_logits {
        ft.data[arg * n_classes + class] = logit;
    }

    let mut source = Tensor::from_vec(&[t, a, 4], vec![0.0; t * a * 4]);
    for k in 0..a {
        source.data[k * 4 + source_class] = 10.0;
    }

    let mut start = Tensor::from_vec(&[t, a, lu], vec![0.0; t * a * lu]);
    let mut end = Tensor::from_vec(&[t, a, lu], vec![0.0; t * a * lu]);
    if let Some((s, e)) = span {
        start.data[arg * lu + s] = 10.0;
        end.data[arg * lu + e] = 10.0;
    }

    let mut tensors = std::collections::HashMap::new();
    tensors.insert("filter_template.logits".into(), ft);
    tensors.insert("source.logits".into(), source);
    tensors.insert("span.start.logits".into(), start);
    tensors.insert("span.end.logits".into(), end);
    HeadOutputs { tensors }
}

/// Decode `pqlFilter` for `utterance`, with the head pointed at `class`.
fn decode_filter(
    utterance: &str,
    class: usize,
    span: Option<(usize, usize)>,
) -> Option<SemanticValue> {
    decode_arg(utterance, &[(class, 10.0)], span, 1, 3)
}

fn decode_arg(
    utterance: &str,
    template_logits: &[(usize, f32)],
    span: Option<(usize, usize)>,
    arg_idx: usize,
    source_class: usize,
) -> Option<SemanticValue> {
    let cfg = tiny_config();
    let tokenizer = NtcTokenizer::from_bytes(test_tokenizer_json().as_bytes()).unwrap();
    let seq = tokenizer.encode_utterance(utterance).unwrap();
    let tool = tool();
    let candidates = [&tool];
    let inputs = ModelInputs::pack(&cfg, &tokenizer, &seq, &candidates).unwrap();
    let tmpl = templates();
    let outs = outputs(&cfg, tmpl.len() + 1, template_logits, span, source_class);

    let decoder = Decoder {
        optional_arg_threshold: 0.0,
        outputs: &outs,
        inputs: &inputs,
        utterance: &seq,
        utterance_text: utterance,
        candidates: &candidates,
        context: None,
        filter_templates: &tmpl,
        action_temperature: 1.0,
        tool_temperature: 1.0,
        presence_temperature: 1.0,
        value_temperature: 1.0,
    };
    decoder
        .value(0, arg_idx, &tool)
        .unwrap()
        .map(|(value, _, _)| value)
}

/// Token indices of the first and last token overlapping `needle`.
fn span_of(utterance: &str, needle: &str) -> (usize, usize) {
    let tokenizer = NtcTokenizer::from_bytes(test_tokenizer_json().as_bytes()).unwrap();
    let seq = tokenizer.encode_utterance(utterance).unwrap();
    let at = utterance.find(needle).expect("needle in utterance");
    let (from, to) = (at, at + needle.len());
    let hits: Vec<usize> = seq
        .offsets
        .iter()
        .enumerate()
        .filter(|(_, &(s, e))| e > s && s < to && e > from)
        .map(|(i, _)| i)
        .collect();
    (hits[0], *hits.last().unwrap())
}

#[test]
fn a_two_slot_template_is_filled_from_one_span() {
    let utterance = "which Teaser have a teaserText below 199";
    let span = span_of(utterance, "teaserText below 199");
    assert_eq!(
        decode_filter(utterance, 2, Some(span)), // class 2 = FIELD_LESS_THAN
        Some(SemanticValue::String("teaserText < 199".into())),
    );
}

#[test]
fn a_constant_template_needs_no_span_at_all() {
    assert_eq!(
        decode_filter("welche Seiten sind noch nicht veroeffentlicht", 3, None),
        Some(SemanticValue::String(
            r#"type = "page" AND published = false"#.into()
        )),
    );
}

#[test]
fn class_zero_means_no_template_and_falls_back_to_the_span() {
    // NONE hands the argument back to the ordinary path, so a TEXT argument
    // binds its span verbatim — the pre-v4 behaviour, unchanged.
    let utterance = "find rows where status is open";
    let span = span_of(utterance, "open");
    assert_eq!(
        decode_arg(utterance, &[(0, 10.0)], Some(span), 1, 0),
        Some(SemanticValue::String("open".into())),
    );
}

#[test]
fn a_template_for_another_annotation_is_unreachable() {
    // SOMETHING_ELSE (class 4) serves SORT.ORDER and carries the highest logit
    // in the row by a wide margin. pqlFilter is FILTER.PQL, so the mask must
    // keep it out entirely: the winner among what is left is NONE, and the
    // argument binds its span. `id DESC` must never appear.
    let utterance = "find rows where status is open";
    let span = span_of(utterance, "open");
    assert_eq!(
        decode_arg(utterance, &[(4, 99.0), (0, 5.0)], Some(span), 1, 0),
        Some(SemanticValue::String("open".into())),
    );
}

#[test]
fn an_argument_with_no_annotation_never_consults_the_head() {
    // className carries no SEMANTIC, so however the template head is pointed
    // it binds its span. Arg 0 with the head pointed at FIELD_LESS_THAN.
    let utterance = "which Teaser have a teaserText below 199";
    let span = span_of(utterance, "Teaser");
    let cfg = tiny_config();
    let tokenizer = NtcTokenizer::from_bytes(test_tokenizer_json().as_bytes()).unwrap();
    let seq = tokenizer.encode_utterance(utterance).unwrap();
    let tool = tool();
    let candidates = [&tool];
    let inputs = ModelInputs::pack(&cfg, &tokenizer, &seq, &candidates).unwrap();
    let tmpl = templates();
    let mut outs = outputs(&cfg, tmpl.len() + 1, &[(2, 10.0)], None, 0);
    // Point arg 0's template row at FIELD_LESS_THAN too, and give it the span.
    let lu = cfg.max_utterance_tokens;
    let n = tmpl.len() + 1;
    outs.tensors.get_mut("filter_template.logits").unwrap().data[2] = 10.0;
    let start = outs.tensors.get_mut("span.start.logits").unwrap();
    start.data[span.0] = 10.0;
    let end = outs.tensors.get_mut("span.end.logits").unwrap();
    end.data[span.1] = 10.0;
    let _ = (lu, n);

    let decoder = Decoder {
        optional_arg_threshold: 0.0,
        outputs: &outs,
        inputs: &inputs,
        utterance: &seq,
        utterance_text: utterance,
        candidates: &candidates,
        context: None,
        filter_templates: &tmpl,
        action_temperature: 1.0,
        tool_temperature: 1.0,
        presence_temperature: 1.0,
        value_temperature: 1.0,
    };
    assert_eq!(
        decoder.value(0, 0, &tool).unwrap().map(|(v, _, _)| v),
        Some(SemanticValue::String("Teaser".into())),
    );
}

#[test]
fn an_unfillable_slot_yields_no_value_rather_than_a_wrong_one() {
    // FIELD_LESS_THAN needs a number and the span holds none. The argument
    // must come back empty — the caller turns that into ASK — instead of
    // binding whatever the span happened to cover.
    let utterance = "which Teaser have a teaserText below 199";
    let span = span_of(utterance, "teaserText");
    assert_eq!(decode_filter(utterance, 2, Some(span)), None);
}

#[test]
fn a_schema_default_fills_an_argument_the_request_never_mentions() {
    // parentId (arg 2) with source Model and a declared `default` of 1.
    // Nothing is invented: the value is read off the schema.
    assert_eq!(
        decode_arg("start at the root of the asset tree", &[(0, 10.0)], None, 2, 3),
        Some(SemanticValue::Integer(1)),
    );
}
