//! Typed Action IR v1 (spec §17–§20, V1 subset per §73).
//!
//! These Rust types are the **source of truth** for the IR contract.
//! `contracts/action-ir/v1/action-ir.schema.json` is generated from them
//! (`ntc-cli gen-schemas`) and the Python side generates pydantic models from
//! that schema. Any change here is a contract change: bump [`crate::IR_VERSION`]
//! and regenerate.
//!
//! Serialization conventions (normative):
//! - enums serialize as SCREAMING_SNAKE_CASE strings,
//! - `SemanticValue` is adjacently tagged: `semantic_type` + `value`,
//!   matching the spec §18 wire shape,
//! - token spans are `[start, end)` **token indices** over the tokenized
//!   utterance; the runtime resolves them to text via tokenizer offsets.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// V1 action states (spec §73; the full §3 set arrives in later phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActionState {
    Call,
    Ask,
    NoCall,
}

/// The typed action program produced by the neural compiler and consumed by
/// the deterministic backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ActionIr {
    /// Always [`crate::IR_VERSION`] for values produced by this crate.
    pub ir_version: u32,
    pub action: ActionState,
    /// Calibrated confidence of the action-state decision, in `[0, 1]`.
    pub action_confidence: f32,
    /// Present iff `action == CALL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<ToolSelection>,
    /// Bound arguments; non-empty only for `CALL`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub arguments: Vec<ArgumentBinding>,
    /// Fields the model could not resolve; drives `ASK`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unresolved: Vec<UnresolvedField>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolSelection {
    /// Index into the candidate set given to the model (0..15 in V1).
    pub candidate_index: u8,
    /// Stable registry id of the selected tool, e.g. `calendar.create`.
    pub registry_id: String,
    /// Calibrated confidence of the tool selection, in `[0, 1]`.
    pub confidence: f32,
}

// NOTE: no `deny_unknown_fields` here — serde does not support it in
// combination with `flatten` (the semantic_type/value pair below).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ArgumentBinding {
    /// Parameter name as declared in the canonical tool schema.
    pub parameter: String,
    #[serde(flatten)]
    pub value: SemanticValue,
    /// Calibrated confidence of this binding, in `[0, 1]`.
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

/// Why an argument is unresolved (V1 subset of spec §16.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UnresolvedReason {
    Missing,
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UnresolvedField {
    pub parameter: String,
    pub reason: UnresolvedReason,
    pub confidence: f32,
}

/// Where a bound value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProvenanceSource {
    /// Extracted from the user utterance (span points into its tokens).
    User,
    /// Resolved from conversational context entities.
    Context,
    /// Inferred by the model without a source span.
    Model,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub source: ProvenanceSource,
    /// `[start, end)` token indices over the tokenized utterance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_span: Option<TokenSpan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenSpan {
    pub start: u32,
    /// Exclusive.
    pub end: u32,
}

// ---------------------------------------------------------------------------
// Semantic values (spec §19, V1 subset)
// ---------------------------------------------------------------------------

/// Fixed vocabulary of relative-date relations. Index assignments (for the
/// datetime head) live in the head codec; this enum is the IR-side vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DateRelation {
    Today,
    Tomorrow,
    Yesterday,
    /// e.g. "this Friday"
    This,
    /// e.g. "next Friday"
    Next,
    /// e.g. "last Friday"
    Last,
    /// e.g. "in 3 days" — magnitude/unit in `offset`
    In,
    /// e.g. "3 days ago" — magnitude/unit in `offset`
    Ago,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Weekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Daypart {
    Morning,
    Noon,
    Afternoon,
    Evening,
    Night,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DurationUnit {
    Second,
    Minute,
    Hour,
    Day,
    Week,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DurationValue {
    pub magnitude: f64,
    pub unit: DurationUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CivilDate {
    pub year: i16,
    /// 1–12
    pub month: u8,
    /// 1–31
    pub day: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CivilTime {
    /// 0–23
    pub hour: u8,
    /// 0–59
    pub minute: u8,
}

/// A semantic value as predicted by the model — richer than JSON types so
/// deterministic code can canonicalize exactly (spec §19).
///
/// Wire shape (adjacent tagging) matches spec §18:
/// `{ "semantic_type": "DURATION", "value": { "magnitude": 1, "unit": "HOUR" } }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "semantic_type", content = "value")]
pub enum SemanticValue {
    #[serde(rename = "STRING")]
    String(String),
    #[serde(rename = "BOOLEAN")]
    Boolean(bool),
    #[serde(rename = "INTEGER")]
    Integer(i64),
    #[serde(rename = "FLOAT")]
    Float(f64),
    /// Index + symbol into the canonical schema's enum-value list for the
    /// bound parameter. `index` is authoritative; `symbol` is redundancy the
    /// validator cross-checks.
    #[serde(rename = "ENUM")]
    Enum { index: u32, symbol: String },
    #[serde(rename = "ABSOLUTE_DATE")]
    AbsoluteDate(CivilDate),
    #[serde(rename = "RELATIVE_DATE")]
    RelativeDate {
        relation: DateRelation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        weekday: Option<Weekday>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<DurationValue>,
    },
    /// RFC 3339 timestamp, e.g. `2026-08-19T15:00:00+02:00`.
    #[serde(rename = "ABSOLUTE_DATETIME")]
    AbsoluteDateTime(String),
    #[serde(rename = "RELATIVE_DATETIME")]
    RelativeDateTime {
        relation: DateRelation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        weekday: Option<Weekday>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        daypart: Option<Daypart>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        time: Option<CivilTime>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<DurationValue>,
    },
    #[serde(rename = "TIME_OF_DAY")]
    TimeOfDay(CivilTime),
    #[serde(rename = "DAYPART")]
    Daypart(Daypart),
    #[serde(rename = "DURATION")]
    Duration(DurationValue),
    /// Person reference by surface text in V1 (no context-entity resolution).
    #[serde(rename = "PERSON_REF")]
    PersonRef { text: String },
    #[serde(rename = "LOCATION")]
    Location { text: String },
}

impl SemanticValue {
    /// The IR-side semantic type name (matches the wire tag).
    pub fn semantic_type_name(&self) -> &'static str {
        match self {
            SemanticValue::String(_) => "STRING",
            SemanticValue::Boolean(_) => "BOOLEAN",
            SemanticValue::Integer(_) => "INTEGER",
            SemanticValue::Float(_) => "FLOAT",
            SemanticValue::Enum { .. } => "ENUM",
            SemanticValue::AbsoluteDate(_) => "ABSOLUTE_DATE",
            SemanticValue::RelativeDate { .. } => "RELATIVE_DATE",
            SemanticValue::AbsoluteDateTime(_) => "ABSOLUTE_DATETIME",
            SemanticValue::RelativeDateTime { .. } => "RELATIVE_DATETIME",
            SemanticValue::TimeOfDay(_) => "TIME_OF_DAY",
            SemanticValue::Daypart(_) => "DAYPART",
            SemanticValue::Duration(_) => "DURATION",
            SemanticValue::PersonRef { .. } => "PERSON_REF",
            SemanticValue::Location { .. } => "LOCATION",
        }
    }
}

// ---------------------------------------------------------------------------
// Compile request (the other half of the runtime boundary)
// ---------------------------------------------------------------------------

/// A known conversational context entity (thin in V1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContextEntity {
    pub id: String,
    /// e.g. PERSON, LOCATION, EVENT
    pub kind: String,
    pub display: String,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequestContext {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<ContextEntity>,
}

/// The runtime's public compile input (spec §12, V1 subset).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompileRequest {
    pub utterance: String,
    /// BCP-47, e.g. `de-DE`. Defaults to the compiler config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    /// IANA timezone, e.g. `Europe/Berlin`. Defaults to the compiler config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
    /// RFC 3339 override of "now" for deterministic resolution in tests/eval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub now: Option<String>,
    /// Registry ids of candidate tools, in candidate order (≤16 in V1).
    /// `None` = use all registered tools (error if more than 16).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<RequestContext>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_round_trip_matches_spec_shape() {
        let ir = ActionIr {
            ir_version: 1,
            action: ActionState::Call,
            action_confidence: 0.998,
            tool: Some(ToolSelection {
                candidate_index: 2,
                registry_id: "calendar.create".into(),
                confidence: 0.997,
            }),
            arguments: vec![
                ArgumentBinding {
                    parameter: "title".into(),
                    value: SemanticValue::String("Zahnarzttermin".into()),
                    confidence: 0.995,
                    provenance: Some(Provenance {
                        source: ProvenanceSource::User,
                        token_span: Some(TokenSpan { start: 7, end: 9 }),
                    }),
                },
                ArgumentBinding {
                    parameter: "start".into(),
                    value: SemanticValue::RelativeDateTime {
                        relation: DateRelation::Tomorrow,
                        weekday: None,
                        daypart: Some(Daypart::Afternoon),
                        time: None,
                        offset: None,
                    },
                    confidence: 0.981,
                    provenance: None,
                },
                ArgumentBinding {
                    parameter: "duration".into(),
                    value: SemanticValue::Duration(DurationValue {
                        magnitude: 1.0,
                        unit: DurationUnit::Hour,
                    }),
                    confidence: 0.994,
                    provenance: None,
                },
            ],
            unresolved: vec![],
        };

        let json = serde_json::to_value(&ir).unwrap();
        // Spec §18 wire shape: adjacent tagging.
        assert_eq!(json["arguments"][0]["semantic_type"], "STRING");
        assert_eq!(json["arguments"][0]["value"], "Zahnarzttermin");
        assert_eq!(json["arguments"][1]["value"]["relation"], "TOMORROW");
        assert_eq!(json["arguments"][1]["value"]["daypart"], "AFTERNOON");
        assert_eq!(json["arguments"][2]["value"]["unit"], "HOUR");
        assert_eq!(json["action"], "CALL");

        let back: ActionIr = serde_json::from_value(json).unwrap();
        assert_eq!(back, ir);
    }

    #[test]
    fn unknown_fields_rejected() {
        let bad = serde_json::json!({
            "ir_version": 1,
            "action": "CALL",
            "action_confidence": 0.9,
            "surprise": true
        });
        assert!(serde_json::from_value::<ActionIr>(bad).is_err());
    }
}
