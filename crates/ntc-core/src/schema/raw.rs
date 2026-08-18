//! Raw tool schema ingestion (spec §39: JSON Schema / OpenAI-function style /
//! flat NTC style). Parsing is deliberately tolerant here; strictness lives in
//! `compile_schema`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::NtcError;
use crate::schema::RiskClass;

/// A raw tool schema as registered by the host application.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawToolSchema {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Either the flat NTC map (`name -> RawParameter`) or a JSON-Schema
    /// object (`{"type": "object", "properties": {...}, "required": [...]}`).
    #[serde(default)]
    pub parameters: Value,
    #[serde(default)]
    pub risk: Option<RiskClass>,
}

/// Flat-style parameter declaration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RawParameter {
    #[serde(rename = "type", default)]
    pub json_type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub required: Option<bool>,
    #[serde(rename = "enum", default)]
    pub enum_values: Option<Vec<Value>>,
    /// Semantic annotation, accepted as `semantic`, `semantic_type`, or
    /// `x-semantic` (aliases handled in [`RawToolSchema::normalized_parameters`]).
    #[serde(default)]
    pub semantic: Option<String>,
    #[serde(rename = "semantic_type", default)]
    pub semantic_type: Option<String>,
    #[serde(rename = "x-semantic", default)]
    pub x_semantic: Option<String>,
}

/// A parameter after style detection, ready for canonical compilation.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedParam {
    pub json_type: String,
    pub description: Option<String>,
    pub format: Option<String>,
    pub required: bool,
    pub enum_values: Vec<String>,
    pub semantic: Option<String>,
}

impl RawToolSchema {
    /// Detect the parameter style and normalize to ordered `(name, param)`
    /// pairs. Order is the declaration order in the JSON document (serde_json
    /// must be built with `preserve_order`; enforced by a unit test).
    pub fn normalized_parameters(&self) -> Result<Vec<(String, NormalizedParam)>, NtcError> {
        let obj = match &self.parameters {
            Value::Null => return Ok(vec![]),
            Value::Object(o) => o,
            other => {
                return Err(NtcError::Schema(format!(
                    "tool `{}`: parameters must be an object, got {other}",
                    self.name
                )))
            }
        };

        // JSON-Schema / OpenAI function style?
        let is_json_schema = obj.get("type").and_then(Value::as_str) == Some("object")
            && obj.get("properties").map(Value::is_object).unwrap_or(false);

        if is_json_schema {
            let props = obj["properties"].as_object().unwrap();
            let required: Vec<&str> = obj
                .get("required")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            props
                .iter()
                .map(|(name, v)| {
                    let p = parse_flat_param(&self.name, name, v)?;
                    Ok((
                        name.clone(),
                        NormalizedParam {
                            required: required.contains(&name.as_str()),
                            ..p
                        },
                    ))
                })
                .collect()
        } else {
            obj.iter()
                .map(|(name, v)| {
                    let p = parse_flat_param(&self.name, name, v)?;
                    Ok((name.clone(), p))
                })
                .collect()
        }
    }
}

fn parse_flat_param(tool: &str, name: &str, v: &Value) -> Result<NormalizedParam, NtcError> {
    let raw: RawParameter = serde_json::from_value(v.clone())
        .map_err(|e| NtcError::Schema(format!("tool `{tool}`, argument `{name}`: {e}")))?;
    let enum_values = raw
        .enum_values
        .unwrap_or_default()
        .into_iter()
        .map(|v| match v {
            Value::String(s) => Ok(s),
            other => Err(NtcError::Schema(format!(
                "tool `{tool}`, argument `{name}`: non-string enum value {other} unsupported in V1"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NormalizedParam {
        json_type: raw.json_type.unwrap_or_else(|| "string".into()),
        description: raw.description,
        format: raw.format,
        required: raw.required.unwrap_or(false),
        enum_values,
        semantic: raw.semantic.or(raw.semantic_type).or(raw.x_semantic),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declaration_order_is_preserved() {
        // The canonical ABI depends on stable argument order; serde_json must
        // be compiled with `preserve_order`.
        let raw: RawToolSchema = serde_json::from_str(
            r#"{"name":"t","parameters":{"zeta":{"type":"string"},"alpha":{"type":"string"},"mid":{"type":"string"}}}"#,
        )
        .unwrap();
        let names: Vec<String> = raw
            .normalized_parameters()
            .unwrap()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, vec!["zeta", "alpha", "mid"]);
    }
}
