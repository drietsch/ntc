//! Deterministic value-template rendering (head codec v4 `filter_template`).
//!
//! Some arguments carry a value the utterance never spells out — a PQL filter,
//! a query expression, a canned predicate. "which BrandAsset have a
//! photographer below 10?" wants `photographer < 10`; that string is nowhere in
//! the request. A compiler with no decoder cannot write it.
//!
//! It does not have to. The *shapes* such values take are a small closed set
//! the host knows, and everything that varies inside them is in the utterance.
//! So the model picks a shape (the `filter_template` head) and marks one span
//! (the span head it already has), and this module fills the shape from the
//! span — the same split the datetime head uses, where the head chooses
//! `NEXT` + `FRIDAY` and deterministic code turns that into a date.
//!
//! Slots, all resolved from the marked span and nothing else:
//!
//! | placeholder | filled with |
//! |---|---|
//! | `{field}`  | the first identifier-shaped word in the span |
//! | `{number}` | the last number in the span |
//! | `{token}`  | the template's declared value the span names |
//!
//! `{token}` exists for slots whose fillers are a closed set the host knows — a
//! file extension, a status word. Matching against that set rather than
//! copying the span is what makes "show me all the PDFs", "welche
//! pdf-dateien haben wir?" and "todos los archivos pdf" all yield `pdf`:
//! plurals, compounds and case never have to be undone, because the answer was
//! never the span's surface in the first place.
//!
//! `{number}` reads the *last* number and both readers stay inside the span on
//! purpose: utterances carry distractor numbers ("list Interaction where
//! optInDate is under 199" has one field and one number, but "get the 3 newest
//! of class Foo under 50" has two). Whatever falls outside the span the model
//! marked is not a candidate, so a correct span is the whole defence — and a
//! slot that cannot be filled returns `None` rather than a guess, which leaves
//! the argument unresolved and turns the call into an ASK.

use crate::normalize::number;

/// Render `pattern` by filling its placeholders from `span_text`.
///
/// Returns `None` if any placeholder the pattern uses has no filler in the
/// span. A pattern with no placeholders is a constant and renders without a
/// span at all.
pub fn render(pattern: &str, span_text: Option<&str>, values: &[String]) -> Option<String> {
    if !pattern.contains('{') {
        return Some(pattern.to_string());
    }
    let text = span_text?;
    let mut out = String::with_capacity(pattern.len() + text.len());
    let mut rest = pattern;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let close = rest[open..].find('}')? + open;
        let filler = match &rest[open + 1..close] {
            "field" => first_identifier(text)?,
            "number" => last_number(text)?,
            "token" => declared_value(text, values)?,
            _ => return None, // unknown placeholder: refuse rather than guess
        };
        out.push_str(&filler);
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    Some(out)
}

/// The first word in the span that looks like a schema field rather than
/// prose: it must contain a letter, must not parse as a number, and must be
/// made only of letters, digits and `_`.
///
/// Surface form is preserved — `precioPromo` and `matchScore` are camelCase in
/// the schema and the utterance quotes them, so lowercasing would break the
/// value. A misspelled field ("liferimeValue") is copied as misspelled; this
/// module will not silently repair it into a different field name.
fn first_identifier(text: &str) -> Option<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .find(|w| {
            !w.is_empty()
                && w.chars().any(char::is_alphabetic)
                && number::parse_number(w).is_none()
        })
        .map(str::to_string)
}

/// The last number in the span, rendered as an integer when it is one.
fn last_number(text: &str) -> Option<String> {
    let n = text
        .split_whitespace()
        .filter_map(|w| number::parse_number(w.trim_matches(|c: char| !c.is_alphanumeric())))
        .next_back()?;
    Some(if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    })
}

/// Which of the template's declared values the span names.
///
/// The span is reduced to bare lowercase alphanumerics and the longest
/// declared value it contains wins, so "PDFs", "pdf-Dateien" and "archivos
/// pdf" all resolve to `pdf` without any language-specific stripping. A span
/// naming nothing declared yields `None` — the host said which values exist
/// and this is not one of them.
///
/// With no declared set the reduced span is the value, which is the right
/// behaviour for an open slot but gives up the plural/compound tolerance.
fn declared_value(text: &str, values: &[String]) -> Option<String> {
    let bare: String = text
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    if bare.is_empty() {
        return None;
    }
    if values.is_empty() {
        return Some(bare);
    }
    values
        .iter()
        .filter(|v| bare.contains(&v.to_lowercase()))
        .max_by_key(|v| v.len())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_pattern_needs_no_span() {
        assert_eq!(
            render(r#"type = "page" AND published = false"#, None, &[]).as_deref(),
            Some(r#"type = "page" AND published = false"#)
        );
    }

    #[test]
    fn fills_field_and_number_from_one_span() {
        assert_eq!(
            render("{field} < {number}", Some("photographer below 10"), &[]).as_deref(),
            Some("photographer < 10")
        );
        assert_eq!(
            render("{field} IS NULL", Some("matchScore"), &[]).as_deref(),
            Some("matchScore IS NULL")
        );
    }

    #[test]
    fn camel_case_field_names_survive() {
        // The schema field is camelCase and the utterance quotes it; folding
        // case here would produce a field the provider does not have.
        assert_eq!(
            render("{field} < {number}", Some("precioPromo por debajo de 99"), &[]).as_deref(),
            Some("precioPromo < 99")
        );
    }

    #[test]
    fn number_slot_takes_the_last_number_in_the_span() {
        // "a score of 85" — the field is fixed by the pattern, the span only
        // has to contain the threshold.
        assert_eq!(
            render("matchScore > {number}", Some("un score de 85"), &[]).as_deref(),
            Some("matchScore > 85")
        );
    }

    #[test]
    fn token_slot_resolves_plurals_compounds_and_case() {
        let exts: Vec<String> = ["pdf", "svg", "xlsx", "csv"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for span in ["pdf", "PDFs", ".PDF", "pdf-Dateien", "archivos pdf"] {
            assert_eq!(
                render(r#"filename LIKE "*.{token}""#, Some(span), &exts).as_deref(),
                Some(r#"filename LIKE "*.pdf""#),
                "span {span:?}"
            );
        }
    }

    #[test]
    fn token_slot_refuses_a_value_the_host_never_declared() {
        let exts: Vec<String> = ["pdf", "csv"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            render(r#"filename LIKE "*.{token}""#, Some("docx"), &exts),
            None
        );
    }

    #[test]
    fn an_unfillable_slot_refuses_rather_than_inventing() {
        // No number in the span → no value. The caller leaves the argument
        // unresolved, which surfaces as ASK, not as a made-up threshold.
        assert_eq!(render("{field} < {number}", Some("photographer"), &[]), None);
        // No span at all, but the pattern needs one.
        assert_eq!(render("{field} IS NULL", None, &[]), None);
        // Digits only → nothing identifier-shaped to bind.
        assert_eq!(render("{field} IS NULL", Some("199"), &[]), None);
    }

    #[test]
    fn unknown_placeholder_is_refused_not_passed_through() {
        assert_eq!(render("{whatever} IS NULL", Some("matchScore"), &[]), None);
    }
}
