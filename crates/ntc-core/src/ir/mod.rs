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

/// V1 action states (spec §73) plus [`ActionState::Delegate`].
///
/// `DELEGATE` is the router's escape hatch: the request is real work, but not
/// work a single typed call can express — a multi-step chain whose later
/// steps depend on earlier *results*, a bulk mutation over an unknown result
/// set, conditional/comparative logic, or open-ended authoring/reasoning.
/// The host hands such utterances to a full LLM agent (which may itself use
/// the same tools). It is deliberately distinct from `NO_CALL` ("nothing to
/// execute here") and from `ASK` ("one missing argument away from a call").
///
/// Models trained before this variant existed emit 3-class action logits;
/// `NtcArchConfig::action_classes` records the width, so old `.ntc` files
/// keep loading and simply never predict `DELEGATE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActionState {
    Call,
    Ask,
    NoCall,
    Delegate,
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
    /// Present iff `action == DELEGATE`: why the router escalated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delegate_reason: Option<DelegateReason>,
    /// Optional hint for the agent: the tool the router believes is involved
    /// (it just could not compile the call itself).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_tool: Option<String>,
    /// Present iff `action == NO_CALL`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_call_reason: Option<NoCallReason>,
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

/// Why a request is handed to a full LLM agent. The router does not just
/// say "not mine" — it says *why*, so the host can route (a payload builder,
/// a batching loop, a planner) and so failures are diagnosable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DelegateReason {
    /// The call needs a nested object/array payload the structured heads
    /// cannot author (`propose_*`, `update_*`).
    PayloadRequired,
    /// A single intent exceeding a hard per-call cap (max 5 elements, max 20
    /// proposals, page-size limits) — needs a loop.
    OverLimit,
    /// Conjunctive or genuinely chained request.
    MultiStep,
    /// The selection spans element types a single-type tool cannot accept.
    MixedElementTypes,
}

/// Why nothing should be executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NoCallReason {
    Chitchat,
    ConceptualQuestion,
    UnsupportedCapability,
    OutOfScope,
    /// The utterance names a tool but only discusses it.
    MentionOnly,
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
    /// Machine-readable cause, so the host can phrase the question
    /// (e.g. `TWO_LINKED_ITEMS_OF_DIFFERENT_TYPE`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// Where a bound value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProvenanceSource {
    #[default]
    /// Extracted from the user utterance (span points into its tokens).
    User,
    /// Bound from an item the user linked into the chat (Studio selection);
    /// `linked_refs` names which one(s).
    LinkedItem,
    /// Resolved by the host's identifier pre-pass; `resolver_token` names the
    /// token that was looked up.
    Resolver,
    /// Resolved from conversational context entities.
    Context,
    /// Constructed by the model with no source in the request (a PQL string,
    /// a default parent id).
    Model,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub source: ProvenanceSource,
    /// `[start, end)` token indices over the tokenized utterance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_span: Option<TokenSpan>,
    /// Refs into `CompileRequest.context.linked` (e.g. `["L1", "L3"]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_refs: Vec<String>,
    /// The identifier token the host's resolver looked up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolver_token: Option<String>,
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

/// Element type of a `LIST` value (scalars only, mirroring the ABI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ListItemType {
    String,
    Integer,
    Float,
    Boolean,
}

/// One element of a `LIST`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ListItem {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
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
    /// Spec §19 `LIST<T>`: a homogeneous list of scalar values.
    ///
    /// Elements are flat values under a single declared `item_type` (matching
    /// the ABI's `ITEM <TYPE>` line) rather than individually tagged values —
    /// a list is homogeneous by construction, so tagging every element
    /// repeats the schema. Elements may come from one span the runtime splits
    /// deterministically, or from several linked items, so `element_provenance`
    /// records per-element origin when it differs from the argument's own.
    #[serde(rename = "LIST")]
    List {
        item_type: ListItemType,
        items: Vec<ListItem>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        element_provenance: Vec<Provenance>,
    },
}

impl ActionIr {
    /// A bare verdict with no tool, arguments or reasons.
    pub fn bare(action: ActionState, confidence: f32) -> Self {
        Self {
            ir_version: crate::IR_VERSION,
            action,
            action_confidence: confidence,
            tool: None,
            arguments: vec![],
            unresolved: vec![],
            delegate_reason: None,
            suggested_tool: None,
            no_call_reason: None,
        }
    }
}

impl UnresolvedField {
    pub fn missing(parameter: impl Into<String>, confidence: f32) -> Self {
        Self {
            parameter: parameter.into(),
            reason: UnresolvedReason::Missing,
            confidence,
            hint: None,
        }
    }

    pub fn ambiguous(parameter: impl Into<String>, confidence: f32) -> Self {
        Self {
            parameter: parameter.into(),
            reason: UnresolvedReason::Ambiguous,
            confidence,
            hint: None,
        }
    }
}

impl Provenance {
    /// A value read out of the utterance at `span`.
    pub fn from_utterance(span: TokenSpan) -> Self {
        Self {
            source: ProvenanceSource::User,
            token_span: Some(span),
            ..Default::default()
        }
    }

    /// A value bound from linked context items (Studio selection).
    pub fn from_linked(refs: Vec<String>) -> Self {
        Self {
            source: ProvenanceSource::LinkedItem,
            linked_refs: refs,
            ..Default::default()
        }
    }

    /// A value the host's identifier pre-pass resolved.
    pub fn from_resolver(token: impl Into<String>) -> Self {
        Self {
            source: ProvenanceSource::Resolver,
            resolver_token: Some(token.into()),
            ..Default::default()
        }
    }
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
            SemanticValue::List { .. } => "LIST",
        }
    }
}

// ---------------------------------------------------------------------------
// Compile request (the other half of the runtime boundary)
// ---------------------------------------------------------------------------

/// An element the user linked into the chat — the host's current selection.
/// Arguments may bind these instead of extracting from the utterance
/// ("tag *this*"), which is what makes an in-application router useful.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LinkedItem {
    /// Stable handle within this request (`L1`, `L2`, …) that argument
    /// provenance points at.
    #[serde(rename = "ref")]
    pub reference: String,
    /// Host element type (`asset`, `document`, `object`, …). Tools disagree
    /// on the symbol for the same thing (`object` vs `data-object`), so the
    /// binding layer maps it per tool rather than assuming one vocabulary.
    #[serde(rename = "type")]
    pub kind: String,
    pub id: i64,
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub is_folder: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_name: Option<String>,
}

/// One candidate interpretation of an identifier-like token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolverCandidate {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: i64,
    #[serde(default)]
    pub key: String,
}

/// The host's identifier pre-pass: integer-like tokens in the utterance,
/// looked up before the model runs. An empty `candidates` list means "not
/// found", which is deliberately indistinguishable from "no permission".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolverEntry {
    pub token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub char_span: Option<CharSpan>,
    #[serde(default)]
    pub candidates: Vec<ResolverCandidate>,
}

/// `[start, end)` character offsets (host-facing; token spans are the
/// model-facing form).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CharSpan {
    pub start: u32,
    pub end: u32,
}

/// A known conversational context entity (generic; hosts without a selection
/// model can use this instead of [`LinkedItem`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ContextEntity {
    pub id: String,
    /// e.g. PERSON, LOCATION, EVENT
    pub kind: String,
    pub display: String,
}

/// Everything the host knows about the moment the utterance was typed.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RequestContext {
    /// Items linked into the chat (the current selection).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked: Vec<LinkedItem>,
    /// Identifier pre-pass results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resolver: Vec<ResolverEntry>,
    /// Total selection size when it exceeds what `linked` enumerates — the
    /// signal that a request is bulk and may exceed a tool's per-call cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_count: Option<u32>,
    /// Where in the host UI the user is (list, detail, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub studio_view: Option<String>,
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
                    provenance: Some(Provenance::from_utterance(TokenSpan { start: 7, end: 9 })),
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
            ..ActionIr::bare(ActionState::Call, 0.0)
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
