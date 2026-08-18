//! Relative date/time resolution (spec §43).
//!
//! The model never learns timezone arithmetic; this module resolves
//! `RELATIVE_DATE`/`RELATIVE_DATETIME` semantics against a pinned "now" in
//! the user's IANA timezone using jiff (DST-correct).
//!
//! Normative conventions (docs/action-ir.md §datetime-resolution):
//! - `THIS(weekday)`  → first occurrence in `[today, today+6]` (today counts),
//! - `NEXT(weekday)`  → first occurrence in `[today+1, today+7]` (strictly future),
//! - `LAST(weekday)`  → first occurrence in `[today-7, today-1]`,
//! - `NEXT`/`LAST` without weekday → +1 / −1 day (same as TOMORROW/YESTERDAY),
//! - daypart clock times come from the (configurable) [`DaypartPolicy`],
//! - an explicit `time` overrides any `daypart`.

use jiff::civil::{Date, DateTime, Time};
use jiff::{tz::TimeZone, Span, Timestamp, Zoned};

use ntc_core::ir::{DateRelation, Daypart, DurationUnit, DurationValue, Weekday};
use ntc_core::NtcError;

/// Injected clock so tests and eval pin "now" (spec §43, deterministic eval).
pub trait Clock {
    fn now(&self) -> Timestamp;
}

/// Real system clock.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        Timestamp::now()
    }
}

/// A fixed instant (tests, eval, `CompileRequest.now`).
pub struct FixedClock(pub Timestamp);

impl Clock for FixedClock {
    fn now(&self) -> Timestamp {
        self.0
    }
}

/// Daypart → local clock time (spec §43 "deployment policy for afternoon").
#[derive(Debug, Clone, PartialEq)]
pub struct DaypartPolicy {
    pub morning: Time,
    pub noon: Time,
    pub afternoon: Time,
    pub evening: Time,
    pub night: Time,
}

impl Default for DaypartPolicy {
    fn default() -> Self {
        Self {
            morning: Time::constant(9, 0, 0, 0),
            noon: Time::constant(12, 0, 0, 0),
            afternoon: Time::constant(15, 0, 0, 0),
            evening: Time::constant(19, 0, 0, 0),
            night: Time::constant(22, 0, 0, 0),
        }
    }
}

impl DaypartPolicy {
    pub fn time_for(&self, daypart: Daypart) -> Time {
        match daypart {
            Daypart::Morning => self.morning,
            Daypart::Noon => self.noon,
            Daypart::Afternoon => self.afternoon,
            Daypart::Evening => self.evening,
            Daypart::Night => self.night,
        }
    }
}

fn jiff_weekday(wd: Weekday) -> jiff::civil::Weekday {
    match wd {
        Weekday::Monday => jiff::civil::Weekday::Monday,
        Weekday::Tuesday => jiff::civil::Weekday::Tuesday,
        Weekday::Wednesday => jiff::civil::Weekday::Wednesday,
        Weekday::Thursday => jiff::civil::Weekday::Thursday,
        Weekday::Friday => jiff::civil::Weekday::Friday,
        Weekday::Saturday => jiff::civil::Weekday::Saturday,
        Weekday::Sunday => jiff::civil::Weekday::Sunday,
    }
}

fn offset_span(offset: &DurationValue) -> Result<Span, NtcError> {
    let m = offset.magnitude;
    if !m.is_finite() || !(0.0..=10_000.0).contains(&m) {
        return Err(NtcError::Normalization(format!(
            "implausible offset magnitude {m}"
        )));
    }
    // Sub-unit fractions resolve through smaller units (1.5 h → 90 min).
    let span = match offset.unit {
        DurationUnit::Second => Span::new().try_seconds(m.round() as i64),
        DurationUnit::Minute => Span::new().try_seconds((m * 60.0).round() as i64),
        DurationUnit::Hour => Span::new().try_seconds((m * 3600.0).round() as i64),
        DurationUnit::Day => Span::new().try_hours((m * 24.0).round() as i64),
        DurationUnit::Week => Span::new().try_hours((m * 24.0 * 7.0).round() as i64),
    };
    span.map_err(|e| NtcError::Normalization(format!("offset out of range: {e}")))
}

/// Resolve the date component of a relative expression against local "now".
fn resolve_date(
    now_local: &Zoned,
    relation: DateRelation,
    weekday: Option<Weekday>,
    offset: Option<&DurationValue>,
) -> Result<Date, NtcError> {
    let today = now_local.date();
    let date = match (relation, weekday) {
        (DateRelation::Today, _) => today,
        (DateRelation::Tomorrow, _) => today.tomorrow().map_err(range_err)?,
        (DateRelation::Yesterday, _) => today.yesterday().map_err(range_err)?,
        (DateRelation::This, Some(wd)) => {
            let target = jiff_weekday(wd);
            let ahead = today.weekday().until(target); // 0..=6
            today
                .checked_add(Span::new().days(ahead as i64))
                .map_err(range_err)?
        }
        (DateRelation::Next, Some(wd)) => {
            let target = jiff_weekday(wd);
            let mut ahead = today.weekday().until(target);
            if ahead == 0 {
                ahead = 7;
            }
            today
                .checked_add(Span::new().days(ahead as i64))
                .map_err(range_err)?
        }
        (DateRelation::Last, Some(wd)) => {
            let target = jiff_weekday(wd);
            let mut back = today.weekday().since(target); // 0..=6
            if back == 0 {
                back = 7;
            }
            today
                .checked_sub(Span::new().days(back as i64))
                .map_err(range_err)?
        }
        (DateRelation::This, None) => today,
        (DateRelation::Next, None) => today.tomorrow().map_err(range_err)?,
        (DateRelation::Last, None) => today.yesterday().map_err(range_err)?,
        (DateRelation::In, _) | (DateRelation::Ago, _) => {
            let off = offset.ok_or_else(|| {
                NtcError::Normalization("IN/AGO relation without an offset".into())
            })?;
            let span = offset_span(off)?;
            let shifted = if relation == DateRelation::In {
                now_local.checked_add(span)
            } else {
                now_local.checked_sub(span)
            }
            .map_err(range_err)?;
            shifted.date()
        }
    };
    Ok(date)
}

fn range_err(e: jiff::Error) -> NtcError {
    NtcError::Normalization(format!("date arithmetic out of range: {e}"))
}

/// Resolve a `RELATIVE_DATE` to a civil date in `tz`.
pub fn resolve_relative_date(
    now: Timestamp,
    tz: &TimeZone,
    relation: DateRelation,
    weekday: Option<Weekday>,
    offset: Option<&DurationValue>,
) -> Result<Date, NtcError> {
    let now_local = now.to_zoned(tz.clone());
    resolve_date(&now_local, relation, weekday, offset)
}

/// Resolve a `RELATIVE_DATETIME` to an RFC 3339 timestamp string in `tz`.
///
/// Time selection: explicit `time` > `daypart` (policy table) > for pure
/// IN/AGO offsets the shifted clock time > 09:00 default.
#[allow(clippy::too_many_arguments)]
pub fn resolve_relative_datetime(
    now: Timestamp,
    tz: &TimeZone,
    policy: &DaypartPolicy,
    relation: DateRelation,
    weekday: Option<Weekday>,
    daypart: Option<Daypart>,
    time: Option<Time>,
    offset: Option<&DurationValue>,
) -> Result<String, NtcError> {
    let now_local = now.to_zoned(tz.clone());

    // Pure clock-shift: IN/AGO with sub-day units keeps the shifted time.
    if matches!(relation, DateRelation::In | DateRelation::Ago) {
        let off = offset
            .ok_or_else(|| NtcError::Normalization("IN/AGO relation without an offset".into()))?;
        if matches!(
            off.unit,
            DurationUnit::Second | DurationUnit::Minute | DurationUnit::Hour
        ) {
            let span = offset_span(off)?;
            let shifted = if relation == DateRelation::In {
                now_local.checked_add(span)
            } else {
                now_local.checked_sub(span)
            }
            .map_err(range_err)?;
            return Ok(shifted.timestamp().to_zoned(tz.clone()).to_string());
        }
    }

    let date = resolve_date(&now_local, relation, weekday, offset)?;
    let clock = time
        .or_else(|| daypart.map(|d| policy.time_for(d)))
        .unwrap_or(Time::constant(9, 0, 0, 0));
    let dt = DateTime::from_parts(date, clock);
    let zoned = tz
        .to_ambiguous_zoned(dt)
        .compatible()
        .map_err(|e| NtcError::Normalization(format!("cannot localize {dt}: {e}")))?;
    Ok(zoned.to_string())
}

/// Format a resolved zoned timestamp as RFC 3339 with offset (no tz name),
/// e.g. `2026-08-19T15:00:00+02:00` — the JSON-facing form (spec §4).
pub fn to_rfc3339(zoned_str: &str) -> String {
    // jiff Zoned Display = RFC 9557, e.g. `...+02:00[Europe/Berlin]`.
    match zoned_str.find('[') {
        Some(i) => zoned_str[..i].to_string(),
        None => zoned_str.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn berlin() -> TimeZone {
        TimeZone::get("Europe/Berlin").unwrap()
    }

    /// Spec §4: 2026-08-18 in Berlin, "morgen Nachmittag", 1 hour →
    /// start = 2026-08-19T15:00:00+02:00.
    #[test]
    fn spec_example_tomorrow_afternoon() {
        let now: Timestamp = "2026-08-18T11:00:00+02:00".parse().unwrap();
        let s = resolve_relative_datetime(
            now,
            &berlin(),
            &DaypartPolicy::default(),
            DateRelation::Tomorrow,
            None,
            Some(Daypart::Afternoon),
            None,
            None,
        )
        .unwrap();
        assert_eq!(to_rfc3339(&s), "2026-08-19T15:00:00+02:00");
    }

    #[test]
    fn next_friday_is_strictly_future() {
        // 2026-08-18 is a Tuesday.
        let now: Timestamp = "2026-08-18T11:00:00+02:00".parse().unwrap();
        let d = resolve_relative_date(
            now,
            &berlin(),
            DateRelation::Next,
            Some(Weekday::Friday),
            None,
        )
        .unwrap();
        assert_eq!(d.to_string(), "2026-08-21");
        // NEXT(Tuesday) from a Tuesday jumps a full week.
        let d = resolve_relative_date(
            now,
            &berlin(),
            DateRelation::Next,
            Some(Weekday::Tuesday),
            None,
        )
        .unwrap();
        assert_eq!(d.to_string(), "2026-08-25");
        // THIS(Tuesday) is today.
        let d = resolve_relative_date(
            now,
            &berlin(),
            DateRelation::This,
            Some(Weekday::Tuesday),
            None,
        )
        .unwrap();
        assert_eq!(d.to_string(), "2026-08-18");
    }

    #[test]
    fn dst_gap_resolves_compatibly() {
        // Europe/Berlin springs forward 2026-03-29 02:00 → 03:00. A morning
        // policy at a gap-adjacent date must still resolve.
        let now: Timestamp = "2026-03-28T12:00:00+01:00".parse().unwrap();
        let policy = DaypartPolicy {
            morning: Time::constant(2, 30, 0, 0), // inside the gap
            ..Default::default()
        };
        let s = resolve_relative_datetime(
            now,
            &berlin(),
            &policy,
            DateRelation::Tomorrow,
            None,
            Some(Daypart::Morning),
            None,
            None,
        )
        .unwrap();
        // Compatible disambiguation pushes forward out of the gap.
        assert!(s.starts_with("2026-03-29T03:30:00+02:00"), "{s}");
    }

    #[test]
    fn in_three_days_keeps_date_semantics() {
        let now: Timestamp = "2026-08-18T23:30:00+02:00".parse().unwrap();
        let d = resolve_relative_date(
            now,
            &berlin(),
            DateRelation::In,
            None,
            Some(&DurationValue {
                magnitude: 3.0,
                unit: DurationUnit::Day,
            }),
        )
        .unwrap();
        assert_eq!(d.to_string(), "2026-08-21");
    }

    #[test]
    fn in_90_minutes_shifts_clock() {
        let now: Timestamp = "2026-08-18T11:00:00+02:00".parse().unwrap();
        let s = resolve_relative_datetime(
            now,
            &berlin(),
            &DaypartPolicy::default(),
            DateRelation::In,
            None,
            None,
            None,
            Some(&DurationValue {
                magnitude: 90.0,
                unit: DurationUnit::Minute,
            }),
        )
        .unwrap();
        assert_eq!(to_rfc3339(&s), "2026-08-18T12:30:00+02:00");
    }
}
