//! Canonical Tool ABI and schema compiler (spec §14, §39–§41).
//!
//! Raw provider schemas (flat NTC style or JSON-Schema/OpenAI function style)
//! are compiled into [`CanonicalTool`] and rendered to the **canonical neural
//! text** the schema encoder consumes. This module is the *only*
//! implementation of that rendering — Python reaches it through the
//! `ntc-schemac` CLI — so training/serving skew is impossible by construction.
//!
//! Rendering grammar (normative; `docs/tool-abi.md`, version [`crate::ABI_VERSION`]):
//!
//! ```text
//! TOOL <candidate_index>
//! DESC <normalized tool description>
//! ARG <k> <name>
//! INFO <normalized arg description>     # only if non-empty
//! TYPE <PARAM_TYPE>
//! REQUIRED 0|1
//! SEMANTIC <SEMANTIC_TYPE>              # only if annotated
//! ENUM <j> <value>                      # one line per enum value
//! ```
//!
//! Determinism rules: text is NFC-normalized, whitespace-collapsed,
//! lowercased, trailing sentence period stripped, capped at
//! [`MAX_DESC_CHARS`] chars; argument order is schema declaration order;
//! lines are `\n`-joined with no trailing newline.

mod raw;

pub use raw::{RawParameter, RawToolSchema};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use crate::error::NtcError;

/// Maximum characters kept from a normalized description.
pub const MAX_DESC_CHARS: usize = 200;

/// Maximum enum values rendered per argument (head codec `E` dimension).
pub const MAX_ENUM_VALUES: usize = 12;

/// Canonical parameter types (spec §14 TYPE line).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ParamType {
    Text,
    Integer,
    Float,
    Boolean,
    Enum,
    Date,
    Datetime,
    Duration,
    Person,
    Location,
}

impl ParamType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ParamType::Text => "TEXT",
            ParamType::Integer => "INTEGER",
            ParamType::Float => "FLOAT",
            ParamType::Boolean => "BOOLEAN",
            ParamType::Enum => "ENUM",
            ParamType::Date => "DATE",
            ParamType::Datetime => "DATETIME",
            ParamType::Duration => "DURATION",
            ParamType::Person => "PERSON",
            ParamType::Location => "LOCATION",
        }
    }
}

/// Optional semantic annotation on an argument (spec §14 SEMANTIC line).
/// Kept as a validated string (dot-separated uppercase path, e.g.
/// `LOCATION.DESTINATION`) rather than a closed enum: annotations are
/// advisory hints for the model, not a runtime dispatch key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct SemanticTypeId(pub String);

/// Risk classification carried through to the policy layer (spec §40, §45).
/// Unused for gating in V1 (no REQUEST_APPROVAL action) but part of the ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RiskClass {
    Read,
    #[default]
    Write,
    Destructive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CanonicalArg {
    pub name: String,
    pub param_type: ParamType,
    /// The provider-facing JSON type this argument serializes to:
    /// `string` | `integer` | `number` | `boolean`.
    pub json_type: String,
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_type: Option<SemanticTypeId>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Enum symbols in canonical order; the enum head's index space and the
    /// IR `ENUM { index }` both point into this list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enum_values: Vec<String>,
}

/// The normalized Tool ABI record (spec §40).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CanonicalTool {
    /// Stable registry id, e.g. `calendar.create`.
    pub id: String,
    /// ABI version this record was compiled with.
    pub abi_version: u32,
    pub description: String,
    #[serde(default)]
    pub risk: RiskClass,
    pub args: Vec<CanonicalArg>,
}

/// Kind of a rendered canonical-text line (mirrors the model's segment-kind
/// vocabulary; see head codec `inputs.packing`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LineKind {
    ToolHeader,
    Desc,
    ArgName,
    Info,
    Type,
    Required,
    Semantic,
    EnumValue,
}

/// One rendered line with byte-offset provenance, for anchor discovery by the
/// input packer.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedLine {
    pub kind: LineKind,
    /// `[start, end)` byte range of the whole line in the rendered text.
    pub range: (usize, usize),
    /// Byte range of the anchor payload: the arg name on `ARG` lines, the
    /// value on `ENUM` lines. `None` for other kinds.
    pub anchor: Option<(usize, usize)>,
    /// Which declared argument this line belongs to (None for tool header/desc).
    pub arg_index: Option<usize>,
    /// Which enum value, for `ENUM` lines.
    pub enum_index: Option<usize>,
}

impl CanonicalTool {
    /// Render the canonical neural text for this tool at `candidate_index`.
    ///
    /// This output is model input; it must be byte-stable across releases for
    /// a given ABI version (golden-tested under `fixtures/schema-abi/`).
    pub fn to_neural_text(&self, candidate_index: usize) -> String {
        self.to_neural_text_with_lines(candidate_index).0
    }

    /// Rendering plus per-line byte offsets and anchor ranges. This is the
    /// single implementation of the rendering grammar — `to_neural_text` is a
    /// projection of it.
    pub fn to_neural_text_with_lines(&self, candidate_index: usize) -> (String, Vec<RenderedLine>) {
        let mut text = String::new();
        let mut lines: Vec<RenderedLine> = Vec::with_capacity(2 + self.args.len() * 4);
        let mut push = |text: &mut String,
                        kind: LineKind,
                        content: String,
                        anchor_rel: Option<(usize, usize)>,
                        arg_index: Option<usize>,
                        enum_index: Option<usize>| {
            if !text.is_empty() {
                text.push('\n');
            }
            let start = text.len();
            text.push_str(&content);
            lines.push(RenderedLine {
                kind,
                range: (start, start + content.len()),
                anchor: anchor_rel.map(|(s, e)| (start + s, start + e)),
                arg_index,
                enum_index,
            });
        };

        push(
            &mut text,
            LineKind::ToolHeader,
            format!("TOOL {candidate_index}"),
            None,
            None,
            None,
        );
        push(
            &mut text,
            LineKind::Desc,
            format!("DESC {}", normalize_text(&self.description)),
            None,
            None,
            None,
        );
        for (k, arg) in self.args.iter().enumerate() {
            let head = format!("ARG {k} ");
            let anchor = (head.len(), head.len() + arg.name.len());
            push(
                &mut text,
                LineKind::ArgName,
                format!("{head}{}", arg.name),
                Some(anchor),
                Some(k),
                None,
            );
            if !arg.description.is_empty() {
                push(
                    &mut text,
                    LineKind::Info,
                    format!("INFO {}", normalize_text(&arg.description)),
                    None,
                    Some(k),
                    None,
                );
            }
            push(
                &mut text,
                LineKind::Type,
                format!("TYPE {}", arg.param_type.as_str()),
                None,
                Some(k),
                None,
            );
            push(
                &mut text,
                LineKind::Required,
                format!("REQUIRED {}", u8::from(arg.required)),
                None,
                Some(k),
                None,
            );
            if let Some(sem) = &arg.semantic_type {
                push(
                    &mut text,
                    LineKind::Semantic,
                    format!("SEMANTIC {}", sem.0),
                    None,
                    Some(k),
                    None,
                );
            }
            for (j, v) in arg.enum_values.iter().enumerate() {
                let norm = normalize_text(v);
                let head = format!("ENUM {j} ");
                let anchor = (head.len(), head.len() + norm.len());
                push(
                    &mut text,
                    LineKind::EnumValue,
                    format!("{head}{norm}"),
                    Some(anchor),
                    Some(k),
                    Some(j),
                );
            }
        }
        (text, lines)
    }

    pub fn arg(&self, name: &str) -> Option<&CanonicalArg> {
        self.args.iter().find(|a| a.name == name)
    }
}

/// Normative text normalization for canonical rendering: NFC, collapse all
/// whitespace runs to single spaces, trim, lowercase, strip one trailing '.',
/// cap at [`MAX_DESC_CHARS`] characters (on a char boundary).
pub fn normalize_text(s: &str) -> String {
    let nfc: String = s.nfc().collect();
    let collapsed = nfc.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut lower = collapsed.to_lowercase();
    if lower.ends_with('.') {
        lower.pop();
    }
    if lower.chars().count() > MAX_DESC_CHARS {
        lower = lower.chars().take(MAX_DESC_CHARS).collect();
        // Avoid a dangling partial word after the cut.
        if let Some(idx) = lower.rfind(' ') {
            lower.truncate(idx);
        }
    }
    lower
}

/// Compile a raw provider schema into the canonical Tool ABI (spec §39).
pub fn compile_schema(raw: &RawToolSchema) -> Result<CanonicalTool, NtcError> {
    if raw.name.trim().is_empty() {
        return Err(NtcError::Schema("tool name must be non-empty".into()));
    }
    let params = raw.normalized_parameters()?;
    let mut args = Vec::with_capacity(params.len());
    for (name, p) in params {
        let has_enum = !p.enum_values.is_empty();
        if p.enum_values.len() > MAX_ENUM_VALUES {
            return Err(NtcError::Schema(format!(
                "argument `{name}`: {} enum values exceeds the V1 limit of {MAX_ENUM_VALUES}",
                p.enum_values.len()
            )));
        }
        let param_type = map_param_type(&p, has_enum, &name)?;
        let json_type = match param_type {
            ParamType::Integer => "integer",
            ParamType::Float => "float",
            ParamType::Boolean => "boolean",
            _ => "string",
        };
        let json_type = match p.json_type.as_str() {
            // Preserve the declared JSON type when it is one of the four
            // provider-facing types (e.g. integer-typed durations).
            t @ ("string" | "integer" | "number" | "boolean") => t.to_string(),
            _ => match json_type {
                "float" => "number".to_string(),
                other => other.to_string(),
            },
        };
        args.push(CanonicalArg {
            name,
            param_type,
            json_type,
            required: p.required,
            semantic_type: p
                .semantic
                .as_deref()
                .map(|s| SemanticTypeId(s.trim().to_uppercase())),
            description: p.description.unwrap_or_default(),
            enum_values: p.enum_values,
        });
    }
    Ok(CanonicalTool {
        id: raw.name.clone(),
        abi_version: crate::ABI_VERSION,
        description: raw.description.clone().unwrap_or_default(),
        risk: raw.risk.unwrap_or_default(),
        args,
    })
}

/// Map raw JSON type + format to the canonical parameter type. Semantic
/// annotations do NOT change the type — spec §14 renders `TYPE TEXT` +
/// `SEMANTIC LOCATION.DESTINATION` as separate facts. `person` / `location` /
/// `duration` are accepted as explicit extended types (internal Rust-style
/// definitions per spec §39).
fn map_param_type(
    p: &raw::NormalizedParam,
    has_enum: bool,
    name: &str,
) -> Result<ParamType, NtcError> {
    if has_enum {
        return Ok(ParamType::Enum);
    }
    match (p.json_type.as_str(), p.format.as_deref()) {
        ("string", Some("date")) | ("date", _) => Ok(ParamType::Date),
        ("string", Some("date-time")) | ("datetime", _) => Ok(ParamType::Datetime),
        ("string", Some("duration")) | ("duration", _) => Ok(ParamType::Duration),
        ("string", _) => Ok(ParamType::Text),
        ("integer", _) => Ok(ParamType::Integer),
        ("number", _) | ("float", _) => Ok(ParamType::Float),
        ("boolean", _) => Ok(ParamType::Boolean),
        ("person", _) => Ok(ParamType::Person),
        ("location", _) => Ok(ParamType::Location),
        (other, _) => Err(NtcError::Schema(format!(
            "argument `{name}`: unsupported type `{other}` in V1"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train_search_raw() -> RawToolSchema {
        serde_json::from_value(serde_json::json!({
            "name": "search_trains",
            "description": "Search for available train journeys.",
            "parameters": {
                "destination_city": {
                    "type": "string",
                    "semantic": "LOCATION.DESTINATION",
                    "required": true
                },
                "departure_time": {
                    "type": "string",
                    "format": "date-time",
                    "required": true
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn spec_section_14_rendering() {
        let tool = compile_schema(&train_search_raw()).unwrap();
        let text = tool.to_neural_text(7);
        let expected = "TOOL 7\n\
                        DESC search for available train journeys\n\
                        ARG 0 destination_city\n\
                        TYPE TEXT\n\
                        REQUIRED 1\n\
                        SEMANTIC LOCATION.DESTINATION\n\
                        ARG 1 departure_time\n\
                        TYPE DATETIME\n\
                        REQUIRED 1";
        assert_eq!(text, expected);
    }

    #[test]
    fn openai_function_style_parses() {
        let raw: RawToolSchema = serde_json::from_value(serde_json::json!({
            "name": "email.send",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Who receives it"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"]}
                },
                "required": ["recipient"]
            }
        }))
        .unwrap();
        let tool = compile_schema(&raw).unwrap();
        assert_eq!(tool.args.len(), 2);
        assert!(tool.arg("recipient").unwrap().required);
        let prio = tool.arg("priority").unwrap();
        assert_eq!(prio.param_type, ParamType::Enum);
        assert!(!prio.required);
        assert_eq!(prio.enum_values, vec!["low", "normal", "high"]);
        let text = tool.to_neural_text(0);
        assert!(text.contains("ENUM 0 low\nENUM 1 normal\nENUM 2 high"));
        assert!(text.contains("INFO who receives it"));
    }

    #[test]
    fn normalization_is_deterministic() {
        assert_eq!(normalize_text("  Créate   an\tEvent.  "), "créate an event");
        // NFC: decomposed é composes to one char.
        assert_eq!(normalize_text("Cre\u{0301}ate"), "créate");
    }
}
