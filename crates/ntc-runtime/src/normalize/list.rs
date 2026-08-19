//! Deterministic list splitting (spec §6.2, §19 `LIST<T>`).
//!
//! The model marks **one span** covering a list region — "42, 55 and 101",
//! "12 und 18", "[42, 55]" — and this module splits it into elements and
//! parses each by the schema's declared item type. No list-specific neural
//! head exists, and none is needed: separators and conjunctions are exact
//! rules, so they belong in deterministic code.

use ntc_core::ir::{ListItem, ListItemType};
use ntc_core::schema::ParamType;

use crate::normalize::number;

/// Coordinating conjunctions for the V1 languages, lowercased.
const CONJUNCTIONS: &[&str] = &[
    "and", "und", "et", "y", "e", "&", "sowie", "ou", "or", "oder", "o",
];

/// Split a span's text into element strings.
///
/// Handles comma/semicolon separated lists, trailing conjunctions
/// ("a, b and c"), bare conjunction lists ("12 und 18"), and JSON-ish
/// brackets/quotes, which are stripped before splitting.
pub fn split_items(text: &str) -> Vec<String> {
    let trimmed = text
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    if trimmed.is_empty() {
        return vec![];
    }

    let mut parts: Vec<String> = trimmed
        .split([',', ';'])
        .flat_map(split_on_conjunction)
        .map(|p| p.trim().trim_matches(['"', '\'']).trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();

    // A single part that never split is still a one-element list.
    if parts.is_empty() && !trimmed.is_empty() {
        parts.push(trimmed.to_string());
    }
    parts
}

/// Split one comma-free chunk on a standalone conjunction word.
fn split_on_conjunction(chunk: &str) -> Vec<String> {
    let words: Vec<&str> = chunk.split_whitespace().collect();
    let mut out: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for w in words {
        if CONJUNCTIONS.contains(&w.to_lowercase().trim_matches('.')) {
            if !current.is_empty() {
                out.push(current.join(" "));
                current.clear();
            }
        } else {
            current.push(w);
        }
    }
    if !current.is_empty() {
        out.push(current.join(" "));
    }
    if out.is_empty() {
        vec![chunk.to_string()]
    } else {
        out
    }
}

/// Parse one element by the declared item type. Returns `None` when the text
/// cannot be interpreted as that type (the caller drops the element and, for
/// required arguments, the policy escalates).
pub fn parse_item(text: &str, item_type: ParamType) -> Option<ListItem> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    match item_type {
        ParamType::Integer => number::parse_number(t).map(|v| ListItem::Integer(v.round() as i64)),
        ParamType::Float => number::parse_number(t).map(ListItem::Float),
        ParamType::Boolean => match t.to_lowercase().as_str() {
            "true" | "yes" | "ja" | "oui" | "sí" | "si" | "1" => Some(ListItem::Boolean(true)),
            "false" | "no" | "nein" | "non" | "0" => Some(ListItem::Boolean(false)),
            _ => None,
        },
        _ => Some(ListItem::String(t.to_string())),
    }
}

/// The IR element type for a schema parameter type.
pub fn list_item_type(item_type: ParamType) -> ListItemType {
    match item_type {
        ParamType::Integer => ListItemType::Integer,
        ParamType::Float => ListItemType::Float,
        ParamType::Boolean => ListItemType::Boolean,
        _ => ListItemType::String,
    }
}

/// Split and parse a span's text into list elements.
pub fn parse_list(text: &str, item_type: ParamType) -> Vec<ListItem> {
    split_items(text)
        .iter()
        .filter_map(|part| parse_item(part, item_type))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_and_conjunction_lists() {
        assert_eq!(split_items("42, 55 and 101"), vec!["42", "55", "101"]);
        assert_eq!(split_items("12 und 18"), vec!["12", "18"]);
        assert_eq!(split_items("42 et 55"), vec!["42", "55"]);
        assert_eq!(split_items("12, 18 y 24"), vec!["12", "18", "24"]);
        assert_eq!(split_items("[42, 55, 101]"), vec!["42", "55", "101"]);
        assert_eq!(split_items("812"), vec!["812"]);
    }

    #[test]
    fn quoted_string_lists() {
        assert_eq!(
            split_items("\"summer\", \"archive\""),
            vec!["summer", "archive"]
        );
    }

    #[test]
    fn parses_by_item_type() {
        assert_eq!(
            parse_list("42, 55 and 101", ParamType::Integer),
            vec![
                ListItem::Integer(42),
                ListItem::Integer(55),
                ListItem::Integer(101)
            ]
        );
        assert_eq!(
            parse_list("summer und archiv", ParamType::Text),
            vec![
                ListItem::String("summer".into()),
                ListItem::String("archiv".into())
            ]
        );
        // Non-numeric text under an INTEGER item type yields no elements.
        assert!(parse_list("hero banner", ParamType::Integer).is_empty());
    }

    #[test]
    fn multilingual_booleans() {
        assert_eq!(
            parse_list("true, nein", ParamType::Boolean),
            vec![ListItem::Boolean(true), ListItem::Boolean(false)]
        );
    }
}
