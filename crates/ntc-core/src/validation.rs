//! IR validation against the canonical Tool ABI (spec §42, §45), reporting
//! issues with spec §64 taxonomy codes.

use serde::{Deserialize, Serialize};

use crate::error::TaxonomyCode;
use crate::ir::{ActionIr, ActionState, SemanticValue};
use crate::schema::{CanonicalTool, ParamType};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub code: TaxonomyCode,
    /// Affected parameter, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    fn push(&mut self, code: TaxonomyCode, parameter: Option<&str>, message: impl Into<String>) {
        self.issues.push(ValidationIssue {
            code,
            parameter: parameter.map(str::to_owned),
            message: message.into(),
        });
    }
}

/// Validate a CALL IR against the selected tool's canonical schema.
/// For ASK/NO_CALL the IR must not carry a tool or arguments.
pub fn validate(ir: &ActionIr, tool: Option<&CanonicalTool>) -> ValidationReport {
    let mut report = ValidationReport::default();

    match ir.action {
        ActionState::Call => {
            let Some(tool) = tool else {
                report.push(
                    TaxonomyCode::E02WrongTool,
                    None,
                    "CALL without a resolvable tool selection",
                );
                return report;
            };
            validate_call(ir, tool, &mut report);
        }
        ActionState::Ask => {
            if ir.unresolved.is_empty() {
                report.push(
                    TaxonomyCode::E12AskedUnnecessarily,
                    None,
                    "ASK with no unresolved fields",
                );
            }
        }
        ActionState::NoCall => {
            if ir.tool.is_some() || !ir.arguments.is_empty() {
                report.push(
                    TaxonomyCode::E01WrongActionState,
                    None,
                    "NO_CALL must not carry a tool or arguments",
                );
            }
        }
    }
    report
}

fn validate_call(ir: &ActionIr, tool: &CanonicalTool, report: &mut ValidationReport) {
    // Required fields present?
    for arg in tool.args.iter().filter(|a| a.required) {
        let bound = ir.arguments.iter().any(|b| b.parameter == arg.name);
        let unresolved = ir.unresolved.iter().any(|u| u.parameter == arg.name);
        if !bound && !unresolved {
            report.push(
                TaxonomyCode::E03MissingRequiredArgument,
                Some(&arg.name),
                format!("required argument `{}` is not bound", arg.name),
            );
        }
    }

    for binding in &ir.arguments {
        let Some(arg) = tool.arg(&binding.parameter) else {
            report.push(
                TaxonomyCode::E04HallucinatedArgument,
                Some(&binding.parameter),
                format!(
                    "argument `{}` does not exist on tool `{}`",
                    binding.parameter, tool.id
                ),
            );
            continue;
        };

        // Type compatibility: which semantic values may bind to which
        // canonical parameter types (normative table, docs/action-ir.md).
        let ok = matches!(
            (&binding.value, arg.param_type),
            (SemanticValue::String(_), ParamType::Text)
                | (SemanticValue::PersonRef { .. }, ParamType::Person)
                | (SemanticValue::PersonRef { .. }, ParamType::Text)
                | (SemanticValue::Location { .. }, ParamType::Location)
                | (SemanticValue::Location { .. }, ParamType::Text)
                | (SemanticValue::Boolean(_), ParamType::Boolean)
                | (SemanticValue::Integer(_), ParamType::Integer)
                | (SemanticValue::Integer(_), ParamType::Float)
                | (SemanticValue::Float(_), ParamType::Float)
                | (SemanticValue::Enum { .. }, ParamType::Enum)
                | (SemanticValue::AbsoluteDate(_), ParamType::Date)
                | (SemanticValue::RelativeDate { .. }, ParamType::Date)
                | (SemanticValue::AbsoluteDateTime(_), ParamType::Datetime)
                | (SemanticValue::RelativeDateTime { .. }, ParamType::Datetime)
                | (SemanticValue::TimeOfDay(_), ParamType::Datetime)
                | (SemanticValue::Duration(_), ParamType::Duration)
                | (SemanticValue::Duration(_), ParamType::Integer)
                | (SemanticValue::Duration(_), ParamType::Float)
        );
        if !ok {
            report.push(
                TaxonomyCode::E07WrongType,
                Some(&binding.parameter),
                format!(
                    "semantic type {} cannot bind to parameter type {:?}",
                    binding.value.semantic_type_name(),
                    arg.param_type
                ),
            );
        }

        // Enum bounds + symbol cross-check.
        if let SemanticValue::Enum { index, symbol } = &binding.value {
            match arg.enum_values.get(*index as usize) {
                None => report.push(
                    TaxonomyCode::E05WrongArgumentValue,
                    Some(&binding.parameter),
                    format!(
                        "enum index {index} out of bounds ({} values)",
                        arg.enum_values.len()
                    ),
                ),
                Some(expected) if expected != symbol => report.push(
                    TaxonomyCode::E05WrongArgumentValue,
                    Some(&binding.parameter),
                    format!("enum symbol `{symbol}` does not match index {index} (`{expected}`)"),
                ),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;
    use crate::schema::{compile_schema, RawToolSchema};

    fn calendar_tool() -> CanonicalTool {
        let raw: RawToolSchema = serde_json::from_value(serde_json::json!({
            "name": "calendar.create",
            "description": "Create a calendar event",
            "parameters": {
                "title": {"type": "string", "required": true},
                "start": {"type": "string", "format": "date-time", "required": true},
                "duration_minutes": {"type": "integer", "semantic": "DURATION"},
                "priority": {"type": "string", "enum": ["low", "normal", "high"]}
            }
        }))
        .unwrap();
        compile_schema(&raw).unwrap()
    }

    fn call_ir(arguments: Vec<ArgumentBinding>) -> ActionIr {
        ActionIr {
            ir_version: 1,
            action: ActionState::Call,
            action_confidence: 0.99,
            tool: Some(ToolSelection {
                candidate_index: 0,
                registry_id: "calendar.create".into(),
                confidence: 0.99,
            }),
            arguments,
            unresolved: vec![],
        }
    }

    fn bind(parameter: &str, value: SemanticValue) -> ArgumentBinding {
        ArgumentBinding {
            parameter: parameter.into(),
            value,
            confidence: 0.9,
            provenance: None,
        }
    }

    #[test]
    fn valid_call_passes() {
        let tool = calendar_tool();
        let ir = call_ir(vec![
            bind("title", SemanticValue::String("Zahnarzttermin".into())),
            bind(
                "start",
                SemanticValue::RelativeDateTime {
                    relation: DateRelation::Tomorrow,
                    weekday: None,
                    daypart: Some(Daypart::Afternoon),
                    time: None,
                    offset: None,
                },
            ),
            bind(
                "duration_minutes",
                SemanticValue::Duration(DurationValue {
                    magnitude: 1.0,
                    unit: DurationUnit::Hour,
                }),
            ),
        ]);
        let report = validate(&ir, Some(&tool));
        assert!(report.is_valid(), "{:?}", report.issues);
    }

    #[test]
    fn missing_required_is_e03() {
        let tool = calendar_tool();
        let ir = call_ir(vec![bind("title", SemanticValue::String("x".into()))]);
        let report = validate(&ir, Some(&tool));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == TaxonomyCode::E03MissingRequiredArgument
                && i.parameter.as_deref() == Some("start")));
    }

    #[test]
    fn hallucinated_arg_is_e04_and_wrong_type_is_e07() {
        let tool = calendar_tool();
        let ir = call_ir(vec![
            bind("title", SemanticValue::String("x".into())),
            bind("start", SemanticValue::Boolean(true)),
            bind("invented", SemanticValue::String("y".into())),
        ]);
        let report = validate(&ir, Some(&tool));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == TaxonomyCode::E07WrongType));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == TaxonomyCode::E04HallucinatedArgument));
    }

    #[test]
    fn enum_symbol_mismatch_is_e05() {
        let tool = calendar_tool();
        let mut ir = call_ir(vec![
            bind("title", SemanticValue::String("x".into())),
            bind(
                "start",
                SemanticValue::AbsoluteDateTime("2026-08-19T15:00:00+02:00".into()),
            ),
            bind(
                "priority",
                SemanticValue::Enum {
                    index: 1,
                    symbol: "high".into(),
                },
            ),
        ]);
        let report = validate(&ir, Some(&tool));
        assert!(report
            .issues
            .iter()
            .any(|i| i.code == TaxonomyCode::E05WrongArgumentValue));

        // Correct symbol passes.
        ir.arguments[2].value = SemanticValue::Enum {
            index: 2,
            symbol: "high".into(),
        };
        assert!(validate(&ir, Some(&tool)).is_valid());
    }
}
