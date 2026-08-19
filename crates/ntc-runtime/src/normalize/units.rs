//! Duration/unit conversion to the schema's target convention (spec §44).
//!
//! The target unit for a DURATION-typed parameter is inferred, in priority
//! order, from: the SEMANTIC annotation (`DURATION.MINUTES`,
//! `DURATION_MINUTES`), the parameter name suffix (`_minutes`, `_seconds`,
//! `_hours`, `_days`), else the default **minutes**. Integer-typed duration
//! parameters always receive a rounded integer.

use ntc_core::ir::{DurationUnit, DurationValue};
use ntc_core::schema::CanonicalArg;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

pub fn target_unit(arg: &CanonicalArg) -> TargetUnit {
    let sem = arg
        .semantic_type
        .as_ref()
        .map(|s| s.0.to_uppercase())
        .unwrap_or_default();
    let name = arg.name.to_lowercase();
    for (needle, unit) in [
        ("SECOND", TargetUnit::Seconds),
        ("MINUTE", TargetUnit::Minutes),
        ("HOUR", TargetUnit::Hours),
        ("DAY", TargetUnit::Days),
    ] {
        if sem.contains(needle) {
            return unit;
        }
    }
    for (suffix, unit) in [
        ("_seconds", TargetUnit::Seconds),
        ("_secs", TargetUnit::Seconds),
        ("_minutes", TargetUnit::Minutes),
        ("_mins", TargetUnit::Minutes),
        ("_hours", TargetUnit::Hours),
        ("_days", TargetUnit::Days),
    ] {
        if name.ends_with(suffix) {
            return unit;
        }
    }
    TargetUnit::Minutes
}

fn to_seconds(v: &DurationValue) -> f64 {
    let per: f64 = match v.unit {
        DurationUnit::Second => 1.0,
        DurationUnit::Minute => 60.0,
        DurationUnit::Hour => 3600.0,
        DurationUnit::Day => 86_400.0,
        DurationUnit::Week => 604_800.0,
    };
    v.magnitude * per
}

/// Convert a semantic duration into the target unit's numeric value.
pub fn convert(value: &DurationValue, target: TargetUnit) -> f64 {
    let secs = to_seconds(value);
    match target {
        TargetUnit::Seconds => secs,
        TargetUnit::Minutes => secs / 60.0,
        TargetUnit::Hours => secs / 3600.0,
        TargetUnit::Days => secs / 86_400.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntc_core::schema::{ParamType, SemanticTypeId};

    fn arg(name: &str, semantic: Option<&str>) -> CanonicalArg {
        CanonicalArg {
            name: name.into(),
            param_type: ParamType::Integer,
            item_type: None,
            json_type: "integer".into(),
            required: false,
            semantic_type: semantic.map(|s| SemanticTypeId(s.into())),
            description: String::new(),
            enum_values: vec![],
        }
    }

    /// Spec §44: "one and a half hours" → duration_minutes: 90.
    #[test]
    fn spec_example_90_minutes() {
        let v = DurationValue {
            magnitude: 1.5,
            unit: DurationUnit::Hour,
        };
        let t = target_unit(&arg("duration_minutes", None));
        assert_eq!(t, TargetUnit::Minutes);
        assert_eq!(convert(&v, t), 90.0);
    }

    #[test]
    fn semantic_annotation_wins() {
        let t = target_unit(&arg("length", Some("DURATION_SECONDS")));
        assert_eq!(t, TargetUnit::Seconds);
        let one_hour = DurationValue {
            magnitude: 1.0,
            unit: DurationUnit::Hour,
        };
        assert_eq!(convert(&one_hour, t), 3600.0);
    }

    #[test]
    fn default_is_minutes() {
        assert_eq!(target_unit(&arg("duration", None)), TargetUnit::Minutes);
    }
}
