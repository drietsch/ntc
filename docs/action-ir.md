# Typed Action IR v1

Source of truth: `crates/ntc-core/src/ir/mod.rs`. Generated schemas:
`contracts/action-ir/v1/` (drift-checked by `ntc gen-schemas --check`).
Fixtures: `fixtures/ir/{valid,invalid}` — Rust serde and Python jsonschema
must accept/reject identically.

## Shape

- `action`: `CALL | ASK | NO_CALL` with `action_confidence`.
- `tool`: `{candidate_index (0..15), registry_id, confidence}` — present for
  CALL (and kept on ASK when the question concerns a selected tool).
- `arguments[]`: `{parameter, semantic_type, value, confidence, provenance?}`
  — the `semantic_type`/`value` pair is adjacently tagged (spec §18 wire
  shape).
- `unresolved[]`: `{parameter, reason: MISSING|AMBIGUOUS, confidence}` —
  non-empty exactly when action is ASK.
- Unknown fields are rejected (`deny_unknown_fields`) except inside
  `arguments` entries (serde flatten limitation); the JSON Schema is the
  stricter check there.

## Semantic types (V1)

STRING, BOOLEAN, INTEGER, FLOAT, ENUM {index, symbol}, ABSOLUTE_DATE
{year, month, day}, RELATIVE_DATE {relation, weekday?, offset?},
ABSOLUTE_DATETIME (RFC 3339 string), RELATIVE_DATETIME {relation, weekday?,
daypart?, time?, offset?}, TIME_OF_DAY {hour, minute}, DAYPART, DURATION
{magnitude, unit}, PERSON_REF {text}, LOCATION {text}, LIST {items: [scalar
SemanticValue, ...]}.

Vocabularies (mirrored in the head codec with a NONE class at index 0):
relations TODAY TOMORROW YESTERDAY THIS NEXT LAST IN AGO (+ABSOLUTE at the
head level), weekdays MONDAY…SUNDAY, dayparts MORNING NOON AFTERNOON EVENING
NIGHT, duration units SECOND MINUTE HOUR DAY WEEK.

## Provenance & spans

`provenance.token_span` is `[start, end)` in **token indices** over the
tokenized utterance. Gold data stores char offsets; the training collator
converts via tokenizer offsets; the runtime resolves back to text through the
offset map (`TokenSeq::span_text`), skipping zero-width special tokens.

## Datetime resolution conventions (deterministic backend)

Implemented in `crates/ntc-runtime/src/normalize/datetime.rs`:

- THIS(wd) → first occurrence in `[today, today+6]` (today counts).
- NEXT(wd) → first occurrence in `[today+1, today+7]` (strictly future).
- LAST(wd) → first occurrence in `[today−7, today−1]`.
- NEXT/LAST without weekday → ±1 day.
- IN/AGO with sub-day units shift the clock; with day/week units they shift
  the date.
- Daypart defaults: MORNING 09:00, NOON 12:00, AFTERNOON 15:00, EVENING
  19:00, NIGHT 22:00 (configurable `DaypartPolicy`); explicit time wins.
- DST gaps/folds resolve with jiff's "compatible" disambiguation.

## Duration serialization

Target unit inferred from SEMANTIC annotation (`DURATION_MINUTES`…), then
parameter-name suffix (`_minutes`, `_hours`…), default **minutes**;
integer-typed parameters get rounded values; string-typed DURATION parameters
get ISO-8601 (`PT90M`). (`crates/ntc-runtime/src/normalize/units.rs`)

## Confidence policy (spec §46 defaults)

CALL below tool threshold (0.35) → NO_CALL; required bindings below 0.30 →
unresolved; CALL with unresolved required fields → ASK; ASK with nothing
unresolved → NO_CALL (fail closed). (`crates/ntc-runtime/src/policy.rs`)
