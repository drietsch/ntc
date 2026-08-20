//! Head-output decoding: logits → typed predictions → Action IR.
//!
//! Implements the decode rules of `contracts/heads/v1/head-spec.json`.
//! All argmax/softmax happens here on CPU — head tensors are tiny.

use ntc_core::ir::{
    ActionIr, ActionState, ArgumentBinding, CivilDate, DateRelation, Daypart, DurationUnit,
    DurationValue, Provenance, ProvenanceSource, SemanticValue, TokenSpan, UnresolvedField,
    Weekday,
};
use ntc_core::schema::{CanonicalArg, CanonicalTool, ParamType};
use ntc_model::config::FilterTemplate;
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

/// Threshold above which an additional linked item joins a multi-item
/// binding ("tag all of these").
const MULTI_LINK_THRESHOLD: f32 = 0.25;

pub struct Decoder<'a> {
    /// Minimum presence confidence for an optional argument to be bound.
    pub optional_arg_threshold: f32,
    pub outputs: &'a HeadOutputs,
    pub inputs: &'a ModelInputs,
    pub utterance: &'a TokenSeq,
    pub utterance_text: &'a str,
    pub candidates: &'a [&'a CanonicalTool],
    /// The request's context frame, if the host supplied one.
    pub context: Option<&'a ntc_core::ir::RequestContext>,
    /// Value templates the model was trained against (head codec v4). Empty
    /// for models without a filter-template head.
    pub filter_templates: &'a [FilterTemplate],
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
    /// Optional head lookup: models predating head-codec v3 do not emit these.
    fn optional(&self, name: &str) -> Option<&ntc_model::Tensor> {
        self.outputs.tensors.get(name)
    }

    /// Why the router escalated (head codec v3; `None` on older models).
    pub fn delegate_reason(&self) -> Option<ntc_core::ir::DelegateReason> {
        use ntc_core::ir::DelegateReason::*;
        let logits = self.optional("delegate_reason.logits")?;
        let (idx, _) = argmax_softmax(&logits.data, 1.0);
        Some(match idx {
            0 => PayloadRequired,
            1 => OverLimit,
            2 => MultiStep,
            _ => MixedElementTypes,
        })
    }

    /// Why nothing should run (head codec v3).
    pub fn no_call_reason(&self) -> Option<ntc_core::ir::NoCallReason> {
        use ntc_core::ir::NoCallReason::*;
        let logits = self.optional("no_call_reason.logits")?;
        let (idx, _) = argmax_softmax(&logits.data, 1.0);
        Some(match idx {
            0 => Chitchat,
            1 => ConceptualQuestion,
            2 => UnsupportedCapability,
            3 => OutOfScope,
            _ => MentionOnly,
        })
    }

    /// Where an argument's value comes from (head codec v3). Defaults to the
    /// utterance so older models keep their span-only behaviour.
    fn value_source(&self, tool_idx: usize, arg_idx: usize) -> ProvenanceSource {
        let Some(t) = self.optional("source.logits") else {
            return ProvenanceSource::User;
        };
        let a = t.shape[1];
        let base = (tool_idx * a + arg_idx) * 4;
        let (idx, _) = argmax_softmax(&t.data[base..base + 4], self.value_temperature);
        match idx {
            0 => ProvenanceSource::User,
            1 => ProvenanceSource::LinkedItem,
            2 => ProvenanceSource::Resolver,
            _ => ProvenanceSource::Model,
        }
    }

    /// Which value template this argument takes, if any (head codec v4).
    ///
    /// Only templates declaring the argument's own `SEMANTIC` annotation
    /// compete; every other class is masked out, exactly as the enum head is
    /// masked to the argument's own enum values. `NONE` (index 0) always
    /// competes and means "no template applies" — the ordinary span/source
    /// path then decides, so an argument that *can* take a template is not
    /// forced into one.
    ///
    /// Returns `None` when the model has no such head, when the argument
    /// carries no matching template, or when `NONE` wins.
    fn filter_template(
        &self,
        tool_idx: usize,
        arg_idx: usize,
        arg: &CanonicalArg,
    ) -> Result<Option<(&'a FilterTemplate, f32)>, NtcError> {
        if self.filter_templates.is_empty() {
            return Ok(None);
        }
        let Some(semantic) = arg.semantic_type.as_ref() else {
            return Ok(None);
        };
        let Some(t) = self.optional("filter_template.logits") else {
            return Ok(None);
        };
        let classes = self.filter_templates.len() + 1;
        let a = t.shape[1];
        let base = (tool_idx * a + arg_idx) * classes;
        let row = &t.data[base..base + classes];

        let mut masked = vec![f32::MIN; classes];
        masked[0] = row[0]; // NONE always competes
        for (i, template) in self.filter_templates.iter().enumerate() {
            if template.semantic == semantic.0 {
                masked[i + 1] = row[i + 1];
            }
        }
        if masked[1..].iter().all(|&v| v == f32::MIN) {
            return Ok(None); // no template serves this argument
        }
        let (idx, confidence) = argmax_softmax(&masked, self.value_temperature);
        Ok((idx > 0).then(|| (&self.filter_templates[idx - 1], confidence)))
    }

    /// Which linked items an argument binds. Multi-select: every item whose
    /// probability clears the threshold, so "tag all of these" binds several.
    fn linked_refs(&self, tool_idx: usize, arg_idx: usize) -> Vec<String> {
        let (Some(t), Some(ctx)) = (self.optional("entity_ref.logits"), self.context) else {
            return vec![];
        };
        let n = ctx.linked.len();
        if n == 0 {
            return vec![];
        }
        let width = t.shape[2];
        let a = t.shape[1];
        let base = (tool_idx * a + arg_idx) * width;
        let row = &t.data[base..base + width];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let none_idx = width - 1;
        let (best, _) = argmax_softmax(row, 1.0);
        if best == none_idx {
            return vec![];
        }
        (0..n.min(width - 1))
            .filter(|&i| sum > 0.0 && exps[i] / sum >= MULTI_LINK_THRESHOLD)
            .map(|i| ctx.linked[i].reference.clone())
            .collect()
    }

    /// Bind a value directly from linked context items.
    fn value_from_linked(
        &self,
        arg: &ntc_core::schema::CanonicalArg,
        refs: &[String],
    ) -> Option<SemanticValue> {
        let ctx = self.context?;
        let items: Vec<&ntc_core::ir::LinkedItem> = refs
            .iter()
            .filter_map(|r| ctx.linked.iter().find(|l| &l.reference == r))
            .collect();
        if items.is_empty() {
            return None;
        }
        match arg.param_type {
            ParamType::Integer => Some(SemanticValue::Integer(items[0].id)),
            ParamType::List => Some(SemanticValue::List {
                item_type: ntc_core::ir::ListItemType::Integer,
                items: items
                    .iter()
                    .map(|i| ntc_core::ir::ListItem::Integer(i.id))
                    .collect(),
                element_provenance: vec![],
            }),
            // Element type as an enum symbol: tools disagree on the vocabulary
            // (`object` vs `data-object`), so match the schema's own list.
            ParamType::Enum => {
                let kind = &items[0].kind;
                arg.enum_values
                    .iter()
                    .position(|v| {
                        v == kind
                            || (kind == "object" && v == "data-object")
                            || (kind == "data-object" && v == "object")
                    })
                    .map(|index| SemanticValue::Enum {
                        index: index as u32,
                        symbol: arg.enum_values[index].clone(),
                    })
            }
            ParamType::Text => Some(SemanticValue::String(items[0].key.clone())),
            _ => None,
        }
    }

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

        // Head codec v4: a value the request does not spell out, chosen from
        // the shapes the host declared. Consulted before anything else,
        // because when a template wins the span is a *slot filler*, not the
        // value — reading it as the value is exactly the failure this head
        // exists to remove. A template whose slots the span cannot fill
        // returns no value at all, which surfaces as ASK rather than as a
        // confidently wrong filter.
        if let Some(chosen) = self.filter_template(tool_idx, arg_idx, arg)? {
            let (template, confidence) = chosen;
            let span_text = if template.pattern.contains('{') {
                self.span(tool_idx, arg_idx)?.2
            } else {
                None
            };
            return Ok(crate::normalize::template::render(
                &template.pattern,
                span_text.as_deref(),
                &template.values,
            )
            .map(|s| (SemanticValue::String(s), confidence, None)));
        }

        // Head codec v3: decide where the value comes from before looking for
        // it. A value bound from the Studio selection has no span at all.
        match self.value_source(tool_idx, arg_idx) {
            ProvenanceSource::LinkedItem => {
                let refs = self.linked_refs(tool_idx, arg_idx);
                if !refs.is_empty() {
                    if let Some(value) = self.value_from_linked(arg, &refs) {
                        return Ok(Some((value, 0.9, Some(Provenance::from_linked(refs)))));
                    }
                }
            }
            ProvenanceSource::Resolver => {
                if let (Some(ctx), Some(text)) = (self.context, self.span(tool_idx, arg_idx)?.2) {
                    if let Some(entry) = ctx.resolver.iter().find(|r| r.token == text.trim()) {
                        if let Some(c) = entry.candidates.first() {
                            let value = match arg.param_type {
                                ParamType::Integer => Some(SemanticValue::Integer(c.id)),
                                ParamType::Text => Some(SemanticValue::String(c.key.clone())),
                                _ => None,
                            };
                            if let Some(v) = value {
                                return Ok(Some((
                                    v,
                                    0.9,
                                    Some(Provenance::from_resolver(entry.token.clone())),
                                )));
                            }
                        }
                    }
                }
            }
            ProvenanceSource::Model => {
                // The model says this value has no source in the request. If
                // the schema declares what the provider uses when the argument
                // is omitted, that is the value — supplying it is reading the
                // schema, not inventing a number. Looking for a span here
                // would be looking for something the model just said is not
                // there.
                if let Some(value) = arg
                    .default_value
                    .as_ref()
                    .and_then(|d| value_from_default(arg, d))
                {
                    return Ok(Some((value, 0.9, None)));
                }
            }
            _ => {}
        }

        let (span, span_conf, span_text) = self.span(tool_idx, arg_idx)?;
        let provenance = Some(Provenance::from_utterance(span));

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

        // A provider that takes a list as one comma-separated string (spec §19
        // tier 1 in TEXT clothing). TYPE stays TEXT — the provider still wants
        // a string — but the span covers a list region ("video-embed and
        // spec-table"), so it is split by the same deterministic rules as
        // `TYPE LIST` and re-joined in the provider's own separator.
        let csv_list = arg
            .semantic_type
            .as_ref()
            .is_some_and(|sem| sem.0 == "LIST.CSV");

        let out = match effective_type {
            ParamType::Text if csv_list => span_text.and_then(|t| {
                let items = crate::normalize::list::split_items(&t);
                (!items.is_empty())
                    .then(|| (SemanticValue::String(items.join(",")), span_conf, provenance))
            }),
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
            // A number the utterance does not contain must not be invented.
            //
            // These arms used to fall back to the asinh magnitude head when no
            // span parsed. That head is *auxiliary* by design (spec §44): the
            // exact value comes from a span and the deterministic parser. Using
            // it as a source produced calls like
            //
            //     "show me the asset"  ->  get_asset(id = 44848)
            //
            // from an utterance with no digits at all — a fabricated entity id,
            // executed against a live MCP server, reported as success. The
            // regression cannot approximate an identifier; there is nothing to
            // approximate. The `provenance: None` it carried was the code
            // stating outright that the value had no source.
            //
            // Returning `None` here routes the argument to `unresolved`, so a
            // required one becomes ASK and an optional one is simply omitted.
            // Asking which asset is always better than fetching an arbitrary
            // one.
            ParamType::Integer => span_text
                .as_deref()
                .and_then(number::parse_number)
                .map(|v| {
                    (
                        SemanticValue::Integer(v.round() as i64),
                        span_conf,
                        provenance,
                    )
                }),
            ParamType::Float => span_text
                .as_deref()
                .and_then(number::parse_number)
                .map(|v| (SemanticValue::Float(v), span_conf, provenance)),
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
                    Some(items) if !items.is_empty() => Some((
                        SemanticValue::List {
                            item_type: crate::normalize::list::list_item_type(item_type),
                            items,
                            element_provenance: vec![],
                        },
                        span_conf,
                        provenance,
                    )),
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
            ..ActionIr::bare(ActionState::Call, 0.0)
        };

        // DELEGATE / NO_CALL are whole-utterance verdicts; attach the reason
        // so the host knows which escalation path applies.
        if ir.action == ActionState::Delegate {
            ir.delegate_reason = self.delegate_reason();
            ir.suggested_tool = decoded.tool.map(|(idx, _)| self.candidates[idx].id.clone());
            return Ok(ir);
        }
        if ir.action == ActionState::NoCall {
            ir.no_call_reason = self.no_call_reason();
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
                // An optional argument the model is only mildly confident
                // about is dropped: including it changes what the call does.
                PresenceState::Present if !arg.required && p_conf < self.optional_arg_threshold => {
                }
                PresenceState::Present => match self.value(tool_idx, k, tool)? {
                    Some((value, v_conf, provenance)) => ir.arguments.push(ArgumentBinding {
                        parameter: arg.name.clone(),
                        value,
                        confidence: (p_conf * v_conf).clamp(0.0, 1.0),
                        provenance,
                    }),
                    None if arg.required => ir
                        .unresolved
                        .push(UnresolvedField::missing(arg.name.clone(), p_conf)),
                    None => {}
                },
                PresenceState::Missing if arg.required => ir
                    .unresolved
                    .push(UnresolvedField::missing(arg.name.clone(), p_conf)),
                PresenceState::Ambiguous if arg.required => ir
                    .unresolved
                    .push(UnresolvedField::ambiguous(arg.name.clone(), p_conf)),
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

/// Convert a schema-declared `default` into a semantic value of the
/// argument's canonical type.
///
/// Deliberately strict: a default that does not match the declared type is
/// dropped rather than coerced, so a sloppy registry cannot smuggle a value
/// past the type system. Enum defaults resolve to their index in
/// `enum_values`; a default naming a symbol the enum does not list is dropped.
fn value_from_default(arg: &CanonicalArg, default: &serde_json::Value) -> Option<SemanticValue> {
    match arg.param_type {
        ParamType::Integer => default.as_i64().map(SemanticValue::Integer),
        ParamType::Float => default.as_f64().map(SemanticValue::Float),
        ParamType::Boolean => default.as_bool().map(SemanticValue::Boolean),
        ParamType::Text | ParamType::Person | ParamType::Location => {
            default.as_str().map(|s| SemanticValue::String(s.to_string()))
        }
        ParamType::Enum => {
            let symbol = default.as_str()?;
            let index = arg.enum_values.iter().position(|v| v == symbol)?;
            Some(SemanticValue::Enum {
                index: index as u32,
                symbol: symbol.to_string(),
            })
        }
        // Dates, durations and composites carry their own decode paths; a
        // literal default for them is not expressible in V1.
        _ => None,
    }
}
