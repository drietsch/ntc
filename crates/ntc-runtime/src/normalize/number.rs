//! Deterministic multilingual number parsing for span texts (V1: EN/DE/FR/ES).
//!
//! Handles numeric literals with either decimal separator, simple number
//! words, and the "one and a half" family. The neural magnitude regression is
//! only the fallback when this parser finds nothing (head codec `numeric`).

/// Parse a number from a span text. Returns `None` when no number is found.
pub fn parse_number(text: &str) -> Option<f64> {
    let lower = text.trim().to_lowercase();
    if lower.is_empty() {
        return None;
    }

    // 1. Numeric literal anywhere in the span ("1.5", "1,5", "90", "2 h").
    if let Some(v) = find_literal(&lower) {
        return Some(v);
    }

    // 2. "X and a half" / "Xeinhalb" family.
    if let Some(v) = half_forms(&lower) {
        return Some(v);
    }

    // 3. Plain number words.
    word_value(&lower).or_else(|| {
        lower
            .split(|c: char| !c.is_alphanumeric())
            .find_map(word_value)
    })
}

fn find_literal(text: &str) -> Option<f64> {
    let mut best: Option<f64> = None;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            let mut seen_sep = false;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || (!seen_sep
                        && (bytes[i] == b'.' || bytes[i] == b',')
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_digit()))
            {
                if bytes[i] == b'.' || bytes[i] == b',' {
                    seen_sep = true;
                }
                i += 1;
            }
            let lit = text[start..i].replace(',', ".");
            if let Ok(v) = lit.parse::<f64>() {
                if best.is_none() {
                    best = Some(v);
                }
            }
        } else {
            i += 1;
        }
    }
    best
}

fn half_forms(text: &str) -> Option<f64> {
    // EN "one and a half", DE "eineinhalb"/"anderthalb"/"zweieinhalb",
    // FR "une heure et demie", ES "una hora y media".
    const DIRECT: &[(&str, f64)] = &[
        ("anderthalb", 1.5),
        ("eineinhalb", 1.5),
        ("zweieinhalb", 2.5),
        ("dreieinhalb", 3.5),
    ];
    for (w, v) in DIRECT {
        if text.contains(w) {
            return Some(*v);
        }
    }
    let has_half = [
        "and a half",
        "und eine halbe",
        "et demi",
        "y media",
        "y medio",
    ]
    .iter()
    .any(|h| text.contains(h));
    if has_half {
        let base = text
            .split_whitespace()
            .find_map(word_value)
            .or_else(|| find_literal(text))
            .unwrap_or(1.0);
        return Some(base + 0.5);
    }
    // Bare "half an hour" / "eine halbe Stunde" / "media hora" / "demi-heure".
    if ["half", "halbe", "media", "demi"]
        .iter()
        .any(|h| text.contains(h))
    {
        return Some(0.5);
    }
    None
}

fn word_value(word: &str) -> Option<f64> {
    let v = match word {
        // EN
        "zero" => 0.0,
        "one" | "a" | "an" => 1.0,
        "two" => 2.0,
        "three" => 3.0,
        "four" => 4.0,
        "five" => 5.0,
        "six" | "seis" => 6.0,
        "seven" | "sieben" => 7.0,
        "eight" | "acht" | "ocho" | "huit" => 8.0,
        "nine" | "neun" | "nueve" | "neuf" => 9.0,
        "ten" | "zehn" | "diez" | "dix" => 10.0,
        // DE
        "null" => 0.0,
        "ein" | "eine" | "eins" | "einen" | "einer" => 1.0,
        "zwei" => 2.0,
        "drei" => 3.0,
        "vier" => 4.0,
        "fünf" | "fuenf" => 5.0,
        "sechs" => 6.0,
        // FR
        "un" | "une" => 1.0,
        "deux" => 2.0,
        "trois" => 3.0,
        "quatre" => 4.0,
        "cinq" => 5.0,
        "sept" => 7.0,
        // ES
        "cero" => 0.0,
        "uno" | "una" => 1.0,
        "dos" => 2.0,
        "tres" => 3.0,
        "cuatro" => 4.0,
        "cinco" => 5.0,
        "siete" => 7.0,
        _ => return None,
    };
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals() {
        assert_eq!(parse_number("90"), Some(90.0));
        assert_eq!(parse_number("1.5"), Some(1.5));
        assert_eq!(parse_number("1,5 Stunden"), Some(1.5));
        assert_eq!(parse_number("in 3 Tagen"), Some(3.0));
    }

    #[test]
    fn words_four_languages() {
        assert_eq!(parse_number("two hours"), Some(2.0));
        assert_eq!(parse_number("eine Stunde"), Some(1.0));
        assert_eq!(parse_number("deux heures"), Some(2.0));
        assert_eq!(parse_number("dos horas"), Some(2.0));
    }

    #[test]
    fn half_family() {
        assert_eq!(parse_number("one and a half hours"), Some(1.5));
        assert_eq!(parse_number("eineinhalb Stunden"), Some(1.5));
        assert_eq!(parse_number("anderthalb Stunden"), Some(1.5));
        assert_eq!(parse_number("una hora y media"), Some(1.5));
        assert_eq!(parse_number("half an hour"), Some(0.5));
        assert_eq!(parse_number("une heure et demie"), Some(1.5));
    }

    #[test]
    fn none_when_absent() {
        assert_eq!(parse_number("dentist appointment"), None);
        assert_eq!(parse_number(""), None);
    }
}
