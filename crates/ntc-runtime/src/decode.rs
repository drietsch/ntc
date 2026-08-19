//! Head-output decoding: logits → typed predictions → Action IR.
//!
//! Implements the decode rules of `contracts/heads/v1/head-spec.json`.
//! All argmax/softmax happens here on CPU — head tensors are tiny.

use ntc_core::ir::{
    ActionIr, ActionState, ArgumentBinding, CivilDate, DateRelation, Daypart, DurationUnit,
    DurationValue, Provenance, ProvenanceSource, SemanticValue, TokenSpan, UnresolvedField,
    UnresolvedReason, Weekday,
};
use ntc_core::schema::{CanonicalTool, ParamType};
use ntc_core::tokenizer::TokenSeq;
use ntc_core::{NtcError, IR_VERSION};
use ntc_model::{HeadOutputs, ModelInputs};

use crate::normalize::number;

/// Maximum span length in tokens (head codec `span.decode`).
pub const MAX_SPAN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceState {
    Present,
    Missing,
    Ambiguous,
    NotApplicable,
}

/// Softmax with temperature over a logits slice; returns (argmax, prob(argmax)).
pub fn argmax_softmax(logits: &[f32], temperature: f32) -> (usize, f32) {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| ((v - max) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    (idx, if sum > 0.0 { exps[idx] / sum } else { 0.0 })
}

pub struct Decoder<'a> {
    pub outputs: &'a HeadOutputs,
    pub inputs: &'a ModelInputs,
    pub utterance: &'a TokenSeq,
    pub utterance_text: &'a str,
    pub candidates: &'a [&'a CanonicalTool],
    pub action_temperature: f32,
    pub tool_temperature: f32,
    pub presence_temperature: f32,
    pub value_temperature: f32,
}

pub struct DecodedAction {
    pub action: ActionState,
    pub action_confidence: f32,
    /// `Some((candidate_index, confidence))`; `None` when NO_TOOL won.
    pub tool: Option<(usize, f32)>,
}

impl<'a> Decoder<'a> {
    pub fn decode_action(&self) -> Result<DecodedAction, NtcError> {
        let action_logits = &self.outputs.get("action.logits")?.data;
        let (a_idx, a_conf) = argmax_softmax(action_logits, self.action_temperature);
        // Head-codec class order is frozen: CALL, ASK, NO_CALL, [DELEGATE].
        // A 3-wide head simply never yields index 3.
        let action = match a_idx {
            0 => ActionState::Call,
            1 => ActionState::Ask,
            2 => ActionState::NoCall,
            _ => ActionState::Delegate,
        };

        let tool_logits = &self.outputs.get("tool.logits")?.data;
        let n = self.candidates.len();
        let (t_idx, t_conf) = argmax_softmax(&tool_logits[..n + 1], self.tool_temperature);
        let tool = if t_idx < n {
            Some((t_idx, t_conf))
        } else {
            None
        };

        Ok(DecodedAction {
            action,
            action_confidence: a_conf,
            tool,
        })
    }

    pub fn presence(
        &self,
        tool_idx: usize,
        arg_idx: usize,
    ) -> Result<(PresenceState, f32), NtcError> {
        let t = self.outputs.get("presence.logits")?;
        let a = t.shape[1];
        let base = (tool_idx * a + arg_idx) * 4;
        let (idx, conf) = argmax_softmax(&t.data[base..base + 4], self.presence_temperature);
        let state = match idx {
            0 => PresenceState::Present,
            1 => PresenceState::Missing,
            2 => PresenceState::Ambiguous,
            _ => PresenceState::NotApplicable,
        };
        Ok((state, conf))
    }

    fn arg_logits(
        &self,
        name: &str,
        tool_idx: usize,
        arg_idx: usize,
        classes: usize,
    ) -> Result<Vec<f32>, NtcError> {
        let t = self.outputs.get(name)?;
        let a = t.shape[1];
        let base = (tool_idx * a + arg_idx) * classes;
        Ok(t.data[base..base + classes].to_vec())
    }

    /// Span decode per head codec: start = argmax; end constrained to
    /// `(start, start+MAX_SPAN]`, exclusive.
    pub fn span(
        &self,
        tool_idx: usize,
        arg_idx: usize,
    ) -> Result<(TokenSpan, f32, Option<String>), NtcError> {
        let lu = self.outputs.get("span.start.logits")?.shape[2];
        let starts = self.arg_logits("span.start.logits", tool_idx, arg_idx, lu)?;
        let ends = self.arg_logits("span.end.logits", tool_idx, arg_idx, lu)?;
        let n = self.inputs.utterance_len;
        let (start, s_conf) = argmax_softmax(&starts[..n], self.value_temperature);
        let window_end = (start + MAX_SPAN).min(n);
        let (rel_end, e_conf) = argmax_softmax(&ends[start..window_end], self.value_temperature);
        let end = start + rel_end + 1; // exclusive
        let text = self
            .utterance
            .span_text(self.utterance_text, start as u32, end as u32)
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_owned);
        Ok((
            TokenSpan {
                start: start as u32,
                end: end as u32,
            },
            s_conf.min(e_conf),
            text,
        ))
    }

    fn scalar_class(
        &self,
        name: &str,
        tool_idx: usize,
        arg_idx: usize,
        classes: usize,
    ) -> Result<(usize, f32), NtcError> {
        let logits = self.arg_logits(name, tool_idx, arg_idx, classes)?;
        Ok(argmax_softmax(&logits, self.value_temperature))
    }

    fn magnitude(&self, tool_idx: usize, arg_idx: usize) -> Result<f64, NtcError> {
        let t = self.outputs.get("numeric.magnitude")?;
        let a = t.shape[1];
        let x = t.data[tool_idx * a + arg_idx] as f64;
        Ok(x.sinh()) // asinh-space regression
    }

    /// Decode one PRESENT argument's value per its canonical parameter type.
    /// Returns `None` when no usable value can be recovered (treated as
    /// unresolved by the caller).
    pub fn value(
        &self,
        tool_idx: usize,
        arg_idx: usize,
        tool: &CanonicalTool,
    ) -> Result<Option<(SemanticValue, f32, Option<Provenance>)>, NtcError> {
        let arg = &tool.args[arg_idx];
        let (span, span_conf, span_text) = self.span(tool_idx, arg_idx)?;
        let provenance = Some(Provenance {
            source: ProvenanceSource::User,
            token_span: Some(span),
        });

        // Spec §40/§44: an INTEGER/FLOAT parameter annotated as a duration
        // (e.g. `duration_minutes` with SEMANTIC DURATION) carries duration
        // semantics — the numeric-unit head decides the unit and the
        // deterministic backend converts to the field's target unit.
        let duration_semantics = arg
            .semantic_type
            .as_ref()
            .is_some_and(|sem| sem.0.contains("DURATION"));
        let effective_type = match arg.param_type {
            ParamType::Integer | ParamType::Float if duration_semantics => ParamType::Duration,
            other => other,
        };

        let out = match effective_type {
            ParamType::Text => span_text.map(|t| (SemanticValue::String(t), span_conf, provenance)),
            ParamType::Person => {
                span_text.map(|t| (SemanticValue::PersonRef { text: t }, span_conf, provenance))
            }
            ParamType::Location => {
                span_text.map(|t| (SemanticValue::Location { text: t }, span_conf, provenance))
            }
            ParamType::Boolean => {
                let (idx, conf) = self.scalar_class("boolean.logits", tool_idx, arg_idx, 2)?;
                Some((SemanticValue::Boolean(idx == 1), conf, None))
            }
            ParamType::Enum => {
                let n = arg.enum_values.len();
                if n == 0 {
                    None
                } else {
                    let logits = self.arg_logits(
                        "enum.logits",
                        tool_idx,
                        arg_idx,
                        self.outputs.get("enum.logits")?.shape[2],
                    )?;
                    let (idx, conf) = argmax_softmax(&logits[..n], self.value_temperature);
                    Some((
                        SemanticValue::Enum {
                            index: idx as u32,
                            symbol: arg.enum_values[idx].clone(),
                        },
                        conf,
                        None,
                    ))
                }
            }
            ParamType::Integer => {
                let parsed = span_text.as_deref().and_then(number::parse_number);
                match parsed {
                    Some(v) => Some((
                        SemanticValue::Integer(v.round() as i64),
                        span_conf,
                        provenance,
                    )),
                    None => Some((
                        SemanticValue::Integer(self.magnitude(tool_idx, arg_idx)?.round() as i64),
                        span_conf * 0.5,
                        None,
                    )),
                }
            }
            ParamType::Float => {
                let parsed = span_text.as_deref().and_then(number::parse_number);
                match parsed {
                    Some(v) => Some((SemanticValue::Float(v), span_conf, provenance)),
                    None => Some((
                        SemanticValue::Float(self.magnitude(tool_idx, arg_idx)?),
                        span_conf * 0.5,
                        None,
                    )),
                }
            }
            ParamType::Duration => {
                let (unit_idx, unit_conf) =
                    self.scalar_class("numeric.unit.logits", tool_idx, arg_idx, 6)?;
                let unit = match unit_idx {
                    1 => Some(DurationUnit::Second),
                    2 => Some(DurationUnit::Minute),
                    3 => Some(DurationUnit::Hour),
                    4 => Some(DurationUnit::Day),
                    5 => Some(DurationUnit::Week),
                    _ => None, // NONE
                };
                let magnitude = span_text
                    .as_deref()
                    .and_then(number::parse_number)
                    .unwrap_or(self.magnitude(tool_idx, arg_idx)?);
                unit.map(|unit| {
                    (
                        SemanticValue::Duration(DurationValue { magnitude, unit }),
                        unit_conf,
                        provenance,
                    )
                })
            }
            ParamType::List => {
                // One span over the list region; deterministic splitting and
                // per-element parsing (spec §6.2) — no list-specific head.
                let item_type = arg.item_type.unwrap_or(ParamType::Text);
                match span_text
                    .as_deref()
                    .map(|t| crate::normalize::list::parse_list(t, item_type))
                {
                    Some(items) if !items.is_empty() => {
                        Some((SemanticValue::List { items }, span_conf, provenance))
                    }
                    _ => None,
                }
            }
            // OPAQUE carries no compilable value; policy routes such tools to
            // DELEGATE before decoding reaches here.
            ParamType::Opaque => None,
            ParamType::Date | ParamType::Datetime => self.decode_datetime(
                tool_idx,
                arg_idx,
                arg.param_type,
                span_text,
                span_conf,
                provenance,
            )?,
        };
        Ok(out)
    }

    fn decode_datetime(
        &self,
        tool_idx: usize,
        arg_idx: usize,
        ptype: ParamType,
        span_text: Option<String>,
        span_conf: f32,
        provenance: Option<Provenance>,
    ) -> Result<Option<(SemanticValue, f32, Option<Provenance>)>, NtcError> {
        let (rel_idx, rel_conf) =
            self.scalar_class("datetime.relation.logits", tool_idx, arg_idx, 10)?;
        let (wd_idx, _) = self.scalar_class("datetime.weekday.logits", tool_idx, arg_idx, 8)?;
        let (dp_idx, _) = self.scalar_class("datetime.daypart.logits", tool_idx, arg_idx, 6)?;

        let weekday = match wd_idx {
            1 => Some(Weekday::Monday),
            2 => Some(Weekday::Tuesday),
            3 => Some(Weekday::Wednesday),
            4 => Some(Weekday::Thursday),
            5 => Some(Weekday::Friday),
            6 => Some(Weekday::Saturday),
            7 => Some(Weekday::Sunday),
            _ => None,
        };
        let daypart = match dp_idx {
            1 => Some(Daypart::Morning),
            2 => Some(Daypart::Noon),
            3 => Some(Daypart::Afternoon),
            4 => Some(Daypart::Evening),
            5 => Some(Daypart::Night),
            _ => None,
        };

        // ABSOLUTE: deterministic parse of the span text.
        if rel_idx == 9 {
            let Some(text) = span_text.as_deref() else {
                return Ok(None);
            };
            return Ok(match ptype {
                ParamType::Date => parse_iso_date(text)
                    .map(|d| (SemanticValue::AbsoluteDate(d), span_conf, provenance)),
                _ => parse_rfc3339_like(text)
                    .map(|s| (SemanticValue::AbsoluteDateTime(s), span_conf, provenance)),
            });
        }

        let relation = match rel_idx {
            1 => DateRelation::Today,
            2 => DateRelation::Tomorrow,
            3 => DateRelation::Yesterday,
            4 => DateRelation::This,
            5 => DateRelation::Next,
            6 => DateRelation::Last,
            7 => DateRelation::In,
            8 => DateRelation::Ago,
            _ => return Ok(None), // NONE: nothing recoverable
        };

        // IN/AGO magnitude+unit from the numeric head.
        let offset = if matches!(relation, DateRelation::In | DateRelation::Ago) {
            let (unit_idx, _) = self.scalar_class("numeric.unit.logits", tool_idx, arg_idx, 6)?;
            let unit = match unit_idx {
                1 => DurationUnit::Second,
                2 => DurationUnit::Minute,
                3 => DurationUnit::Hour,
                4 => DurationUnit::Day,
                5 => DurationUnit::Week,
                _ => DurationUnit::Day,
            };
            let magnitude = span_text
                .as_deref()
                .and_then(number::parse_number)
                .unwrap_or(self.magnitude(tool_idx, arg_idx)?);
            Some(DurationValue { magnitude, unit })
        } else {
            None
        };

        let value = match ptype {
            ParamType::Date => SemanticValue::RelativeDate {
                relation,
                weekday,
                offset,
            },
            _ => SemanticValue::RelativeDateTime {
                relation,
                weekday,
                daypart,
                time: None,
                offset,
            },
        };
        Ok(Some((value, rel_conf, provenance)))
    }

    /// Full decode: action/tool + per-arg presence and values → raw IR
    /// (policy adjustments happen afterwards in [`crate::policy`]).
    pub fn decode(&self) -> Result<ActionIr, NtcError> {
        let decoded = self.decode_action()?;
        let mut ir = ActionIr {
            ir_version: IR_VERSION,
            action: decoded.action,
            action_confidence: decoded.action_confidence,
            tool: None,
            arguments: vec![],
            unresolved: vec![],
        };

        // DELEGATE is a whole-utterance verdict: no tool, no arguments.
        if ir.action == ActionState::Delegate {
            return Ok(ir);
        }

        let Some((tool_idx, tool_conf)) = decoded.tool else {
            // NO_TOOL selected: a CALL cannot stand.
            if ir.action == ActionState::Call {
                ir.action = ActionState::NoCall;
            }
            return Ok(ir);
        };
        let tool = self.candidates[tool_idx];

        // Bind arguments for CALL/ASK decoding (ASK needs unresolved info).
        for (k, arg) in tool.args.iter().enumerate() {
            let (presence, p_conf) = self.presence(tool_idx, k)?;
            match presence {
                PresenceState::Present => match self.value(tool_idx, k, tool)? {
                    Some((value, v_conf, provenance)) => ir.arguments.push(ArgumentBinding {
                        parameter: arg.name.clone(),
                        value,
                        confidence: (p_conf * v_conf).clamp(0.0, 1.0),
                        provenance,
                    }),
                    None if arg.required => ir.unresolved.push(UnresolvedField {
                        parameter: arg.name.clone(),
                        reason: UnresolvedReason::Missing,
                        confidence: p_conf,
                    }),
                    None => {}
                },
                PresenceState::Missing if arg.required => ir.unresolved.push(UnresolvedField {
                    parameter: arg.name.clone(),
                    reason: UnresolvedReason::Missing,
                    confidence: p_conf,
                }),
                PresenceState::Ambiguous if arg.required => ir.unresolved.push(UnresolvedField {
                    parameter: arg.name.clone(),
                    reason: UnresolvedReason::Ambiguous,
                    confidence: p_conf,
                }),
                _ => {}
            }
        }

        ir.tool = Some(ntc_core::ir::ToolSelection {
            candidate_index: tool_idx as u8,
            registry_id: tool.id.clone(),
            confidence: tool_conf,
        });
        Ok(ir)
    }
}

fn parse_iso_date(text: &str) -> Option<CivilDate> {
    let t = text.trim();
    let parts: Vec<&str> = t.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    Some(CivilDate {
        year: parts[0].parse().ok()?,
        month: parts[1].parse().ok()?,
        day: parts[2].parse().ok()?,
    })
}

fn parse_rfc3339_like(text: &str) -> Option<String> {
    let t = text.trim();
    // Delegate real validation to jiff at normalization time; here we only
    // check plausibility (date part + 'T').
    if t.len() >= 10 && t.as_bytes().get(4) == Some(&b'-') {
        Some(t.to_string())
    } else {
        None
    }
}
