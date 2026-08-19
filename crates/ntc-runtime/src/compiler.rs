//! `NeuralToolCompiler` (spec §70): tokenize → retrieve candidates → pack →
//! infer → decode → policy → validate → normalize → serialize.

use jiff::tz::TimeZone;
use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use ntc_core::ir::{ActionIr, ActionState, CompileRequest, SemanticValue, UnresolvedField};
use ntc_core::schema::{CanonicalArg, CanonicalTool, ParamType, RawToolSchema};
use ntc_core::tokenizer::NtcTokenizer;
use ntc_core::validation::validate;
use ntc_core::{NtcError, ToolId, ToolRegistry};
use ntc_format::NtcFile;
use ntc_model::{Backend, CpuRefBackend, ModelInputs, ModelWeights, NtcArchConfig};

use crate::decode::Decoder;
use crate::normalize::datetime::{
    resolve_relative_date, resolve_relative_datetime, to_rfc3339, DaypartPolicy,
};
use crate::normalize::units;
use crate::policy::ConfidencePolicy;

#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// BCP-47 default locale, e.g. `de-DE`.
    pub locale: String,
    /// IANA default timezone, e.g. `Europe/Berlin`.
    pub timezone: String,
    pub daypart_policy: DaypartPolicy,
    pub confidence: ConfidencePolicy,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        let mut confidence = ConfidencePolicy::default();
        // Calibrating this threshold needs a sweep over a dev set, so allow
        // an override without a rebuild. Hosts set it through the API.
        if let Ok(v) = std::env::var("NTC_OPTIONAL_ARG_THRESHOLD") {
            if let Ok(v) = v.parse::<f32>() {
                confidence.optional_arg_threshold = v;
            }
        }
        Self {
            locale: "en-US".into(),
            timezone: "UTC".into(),
            daypart_policy: DaypartPolicy::default(),
            confidence,
        }
    }
}

/// What a shortlist round decided, and on what evidence.
///
/// Returned by [`NeuralToolCompiler::shortlist`] for callers that want to see
/// or override the narrowing; `compile` uses only `kept`.
#[derive(Debug, Clone)]
pub struct Shortlist {
    /// Tools offered before narrowing.
    pub considered: usize,
    /// Forward passes the narrowing cost.
    pub rounds: usize,
    /// Every tool with its margin over NO_TOOL, best first. Kept so a host can
    /// tell "one clear winner" from "eight tools within noise of each other",
    /// which is a genuine ASK signal rather than a coin flip.
    pub scores: Vec<(ToolId, f32)>,
    /// The slate handed to the deciding pass.
    pub kept: Vec<ToolId>,
}

/// A validated, executable tool call (spec §4 backend output).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CompiledCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// The compiler's public result (spec §3, V1 action subset).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "outcome", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CompileOutcome {
    Call {
        call: CompiledCall,
        ir: ActionIr,
    },
    Ask {
        unresolved: Vec<UnresolvedField>,
        ir: ActionIr,
    },
    NoCall {
        ir: ActionIr,
    },
    /// Beyond a single typed call — the host should hand `utterance` to a
    /// full LLM agent (which may use the same tools over several turns).
    Delegate {
        utterance: String,
        /// Registry ids offered to the router, so the host can pass the same
        /// candidate set to the agent.
        candidates: Vec<String>,
        ir: ActionIr,
    },
}

pub struct NeuralToolCompiler<B: Backend> {
    arch: NtcArchConfig,
    tokenizer: NtcTokenizer,
    registry: ToolRegistry,
    backend: B,
    config: CompilerConfig,
}

impl NeuralToolCompiler<CpuRefBackend> {
    /// Load a `.ntc` model and run it on the CPU reference backend.
    /// (GPU construction lives in `ntc-webgpu` via [`Self::from_parts`].)
    pub fn load_cpu(model_bytes: &[u8], config: CompilerConfig) -> Result<Self, NtcError> {
        let file = NtcFile::parse(model_bytes).map_err(|e| NtcError::Format(e.to_string()))?;
        if file.metadata.architecture != ntc_model::ARCHITECTURE {
            return Err(NtcError::Format(format!(
                "unsupported architecture `{}` (runtime implements `{}`)",
                file.metadata.architecture,
                ntc_model::ARCHITECTURE
            )));
        }
        let arch = NtcArchConfig::from_metadata(&file.metadata.model)?;
        let weights = ModelWeights::from_ntc(&file, &arch)?;
        let tokenizer = NtcTokenizer::from_bytes(file.tokenizer_bytes)?;
        let backend = CpuRefBackend::new(arch.clone(), weights);
        Ok(Self::from_parts(arch, tokenizer, backend, config))
    }
}

impl<B: Backend> NeuralToolCompiler<B> {
    pub fn from_parts(
        arch: NtcArchConfig,
        tokenizer: NtcTokenizer,
        backend: B,
        config: CompilerConfig,
    ) -> Self {
        Self {
            arch,
            tokenizer,
            registry: ToolRegistry::new(),
            backend,
            config,
        }
    }

    pub fn register_tool(&mut self, schema: RawToolSchema) -> Result<ToolId, NtcError> {
        self.registry.register(schema)
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    pub fn arch(&self) -> &NtcArchConfig {
        &self.arch
    }

    /// Run only the neural stage and return raw head logits as JSON
    /// (`{name: {shape, data}}`) — the parity/debug hook consumed by
    /// `ntc infer --dump-heads` and `eval/parity` (docs/parity-testing.md).
    pub fn run_heads(&mut self, req: &CompileRequest) -> Result<serde_json::Value, NtcError> {
        let candidate_ids = self
            .registry
            .resolve_candidates(req.candidates.as_deref())?;
        let candidates: Vec<&CanonicalTool> = candidate_ids
            .iter()
            .map(|&id| self.registry.get(id).expect("resolved id exists"))
            .collect();
        let utterance = self.tokenizer.encode_utterance(&req.utterance)?;
        let inputs = ModelInputs::pack(&self.arch, &self.tokenizer, &utterance, &candidates)?;
        let outputs = self.backend.run(&inputs)?;
        let mut map = serde_json::Map::new();
        let mut names: Vec<_> = outputs.tensors.keys().collect();
        names.sort();
        for name in names {
            let t = &outputs.tensors[name];
            map.insert(
                name.clone(),
                serde_json::json!({"shape": t.shape, "data": t.data}),
            );
        }
        Ok(serde_json::Value::Object(map))
    }

    /// Steps 1–2 of the pipeline: candidate resolution, tokenization,
    /// packing. Returns owned data so the backend call can borrow `self`
    /// mutably afterwards.
    fn prepare(
        &self,
        req: &CompileRequest,
    ) -> Result<
        (
            Vec<ntc_core::ToolId>,
            ntc_core::tokenizer::TokenSeq,
            ModelInputs,
        ),
        NtcError,
    > {
        let candidate_ids = self
            .registry
            .resolve_candidates(req.candidates.as_deref())?;
        let (utterance, inputs) = self.prepare_slate(req, &candidate_ids)?;
        Ok((candidate_ids, utterance, inputs))
    }

    /// Pack one explicit slate. Split out from [`Self::prepare`] so the
    /// shortlist rounds can pack a slate the caller never asked for.
    fn prepare_slate(
        &self,
        req: &CompileRequest,
        candidate_ids: &[ntc_core::ToolId],
    ) -> Result<(ntc_core::tokenizer::TokenSeq, ModelInputs), NtcError> {
        let limit = self.slate_limit();
        if candidate_ids.len() > limit {
            return Err(NtcError::CandidateLimit(format!(
                "{} candidates exceeds the model limit {limit}",
                candidate_ids.len()
            )));
        }
        let candidates: Vec<&CanonicalTool> = candidate_ids
            .iter()
            .map(|&id| self.registry.get(id).expect("resolved id exists"))
            .collect();
        let utterance = self.tokenizer.encode_utterance(&req.utterance)?;
        let linked = req
            .context
            .as_ref()
            .map(|c| c.linked.as_slice())
            .unwrap_or(&[]);
        let inputs = ModelInputs::pack_with_context(
            &self.arch,
            &self.tokenizer,
            &utterance,
            &candidates,
            linked,
        )?;
        Ok((utterance, inputs))
    }

    /// How many tools fit in one forward pass.
    fn slate_limit(&self) -> usize {
        self.arch.max_tools.min(ntc_core::registry::MAX_SLATE)
    }

    /// Steps 4–6: decode, policy, validation, serialization.
    fn postprocess(
        &self,
        req: &CompileRequest,
        candidate_ids: &[ntc_core::ToolId],
        utterance: &ntc_core::tokenizer::TokenSeq,
        inputs: &ModelInputs,
        outputs: &ntc_model::HeadOutputs,
    ) -> Result<CompileOutcome, NtcError> {
        let candidates: Vec<&CanonicalTool> = candidate_ids
            .iter()
            .map(|&id| self.registry.get(id).expect("resolved id exists"))
            .collect();
        let cal = &self.arch.calibration;
        let decoder = Decoder {
            outputs,
            inputs,
            utterance,
            utterance_text: &req.utterance,
            candidates: &candidates,
            context: req.context.as_ref(),
            action_temperature: cal.action,
            tool_temperature: cal.tool,
            presence_temperature: cal.presence,
            value_temperature: cal.value,
            optional_arg_threshold: self.config.confidence.optional_arg_threshold,
        };
        let mut ir = decoder.decode()?;

        let selected = ir
            .tool
            .as_ref()
            .map(|t| candidates[t.candidate_index as usize]);
        self.config.confidence.apply(&mut ir, selected);
        let selected = ir
            .tool
            .as_ref()
            .map(|t| candidates[t.candidate_index as usize]);

        match ir.action {
            ActionState::Call => {
                let tool = selected.expect("policy guarantees a tool for CALL");
                let report = validate(&ir, Some(tool));
                if !report.is_valid() {
                    return Err(NtcError::Validation(report.issues));
                }
                let call = self.serialize_call(&ir, tool, req)?;
                Ok(CompileOutcome::Call { call, ir })
            }
            ActionState::Ask => Ok(CompileOutcome::Ask {
                unresolved: ir.unresolved.clone(),
                ir,
            }),
            ActionState::NoCall => Ok(CompileOutcome::NoCall { ir }),
            ActionState::Delegate => Ok(CompileOutcome::Delegate {
                utterance: req.utterance.clone(),
                candidates: candidates.iter().map(|t| t.id.clone()).collect(),
                ir,
            }),
        }
    }

    /// Compile one request.
    ///
    /// When the caller offers more tools than fit in a forward pass, this
    /// runs a **shortlist round** first (see [`Self::shortlist`]) and decides
    /// over the survivors. A slate that already fits is compiled directly, so
    /// the single-pass path is unchanged.
    pub fn compile(&mut self, req: &CompileRequest) -> Result<CompileOutcome, NtcError> {
        let ids = self
            .registry
            .resolve_candidates(req.candidates.as_deref())?;
        let ids = if ids.len() > self.slate_limit() {
            self.shortlist(req, &ids)?.kept
        } else {
            ids
        };
        self.decide(req, &ids)
    }

    /// One forward pass over an already-narrow slate, decoded to an outcome.
    fn decide(&mut self, req: &CompileRequest, ids: &[ToolId]) -> Result<CompileOutcome, NtcError> {
        let (utterance, inputs) = self.prepare_slate(req, ids)?;
        let outputs = self.backend.run(&inputs)?;
        self.postprocess(req, ids, &utterance, &inputs, &outputs)
    }

    /// Narrow a wide tool set down to one slate, then let [`Self::decide`]
    /// re-evaluate the survivors side by side.
    ///
    /// A real MCP host registers every tool it has — Pimcore Studio offers 49
    /// — while the model reads a fixed-width slate. Something has to choose,
    /// and "whatever the caller passed" is not a strategy: it silently makes
    /// the host responsible for the accuracy of the router.
    ///
    /// So the tool set is split into slates and each is scored, keeping the
    /// per-tool margin **against that slate's own NO_TOOL logit**. NO_TOOL is
    /// the one option present in every round, which makes it a usable common
    /// reference; raw logits and per-slate softmax probabilities are not
    /// comparable across rounds, because a slate of three strong candidates
    /// splits its mass while a slate of three decoys does not.
    ///
    /// Rounds are independent, so the survivors have never been *compared* —
    /// only ranked against a shared baseline. That is what the deciding pass
    /// is for: the fusion stack attends across the tools in a slate, which is
    /// how near-identical siblings (`get_asset` / `list_assets` /
    /// `search_assets`) get discriminated. Seeing them together is exactly the
    /// case the model was trained on.
    pub fn shortlist(
        &mut self,
        req: &CompileRequest,
        ids: &[ToolId],
    ) -> Result<Shortlist, NtcError> {
        let width = self.slate_limit();
        let mut scored: Vec<(ToolId, f32)> = Vec::with_capacity(ids.len());
        let mut rounds = 0;

        for chunk in ids.chunks(width) {
            let (_, inputs) = self.prepare_slate(req, chunk)?;
            let outputs = self.backend.run(&inputs)?;
            rounds += 1;

            let logits = &outputs.get("tool.logits")?.data;
            // `[candidate_0 .. candidate_{n-1}, NO_TOOL]`, sized to the slate
            // actually packed — not padded to the model width — so NO_TOOL
            // sits at this chunk's length. A short final chunk is therefore
            // scored against a shorter slate, which is fine: the margin is
            // taken against that same slate's own NO_TOOL.
            let no_tool = logits[chunk.len()];
            for (i, id) in chunk.iter().enumerate() {
                scored.push((id.clone(), logits[i] - no_tool));
            }
        }

        // Ties are broken by the registry order the caller gave, so the same
        // request always shortlists the same tools.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let kept: Vec<ToolId> = scored
            .iter()
            .take(width)
            .map(|(id, _)| id.clone())
            .collect();
        Ok(Shortlist {
            considered: ids.len(),
            rounds,
            scores: scored,
            kept,
        })
    }

    fn now_and_tz(&self, req: &CompileRequest) -> Result<(Timestamp, TimeZone), NtcError> {
        let now: Timestamp = match &req.now {
            Some(s) => s
                .parse()
                .map_err(|e| NtcError::Normalization(format!("bad `now` override: {e}")))?,
            None => Timestamp::now(),
        };
        let tz_name = req.timezone.as_deref().unwrap_or(&self.config.timezone);
        let tz = TimeZone::get(tz_name)
            .map_err(|e| NtcError::Normalization(format!("unknown timezone `{tz_name}`: {e}")))?;
        Ok((now, tz))
    }

    /// Deterministic backend (spec §42): resolve semantics → JSON arguments.
    fn serialize_call(
        &self,
        ir: &ActionIr,
        tool: &CanonicalTool,
        req: &CompileRequest,
    ) -> Result<CompiledCall, NtcError> {
        let (now, tz) = self.now_and_tz(req)?;
        let mut args = serde_json::Map::new();
        for binding in &ir.arguments {
            let arg = tool
                .arg(&binding.parameter)
                .expect("validated: binding exists on tool");
            let value = self.json_value(&binding.value, arg, now, &tz)?;
            // Flattened object properties (`data.key`) re-nest into the
            // object the provider schema declared (schema compiler tier 2).
            match binding.parameter.split_once('.') {
                Some((parent, child)) => {
                    let entry = args
                        .entry(parent.to_string())
                        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                    if let Some(obj) = entry.as_object_mut() {
                        obj.insert(child.to_string(), value);
                    }
                }
                None => {
                    args.insert(binding.parameter.clone(), value);
                }
            }
        }
        Ok(CompiledCall {
            name: tool.id.clone(),
            arguments: serde_json::Value::Object(args),
        })
    }

    fn json_value(
        &self,
        value: &SemanticValue,
        arg: &CanonicalArg,
        now: Timestamp,
        tz: &TimeZone,
    ) -> Result<serde_json::Value, NtcError> {
        use serde_json::json;
        let policy = &self.config.daypart_policy;
        let v = match value {
            SemanticValue::String(s)
            | SemanticValue::PersonRef { text: s }
            | SemanticValue::Location { text: s } => json!(s),
            SemanticValue::Boolean(b) => json!(b),
            SemanticValue::Integer(i) => json!(i),
            SemanticValue::Float(f) => json!(f),
            SemanticValue::Enum { symbol, .. } => json!(symbol),
            SemanticValue::AbsoluteDate(d) => {
                json!(format!("{:04}-{:02}-{:02}", d.year, d.month, d.day))
            }
            SemanticValue::AbsoluteDateTime(s) => {
                // Validate through jiff; emit RFC 3339 in the request tz when
                // the input has no offset.
                if let Ok(ts) = s.parse::<Timestamp>() {
                    json!(to_rfc3339(&ts.to_zoned(tz.clone()).to_string()))
                } else if let Ok(dt) = s.parse::<jiff::civil::DateTime>() {
                    let zoned = tz
                        .to_ambiguous_zoned(dt)
                        .compatible()
                        .map_err(|e| NtcError::Normalization(e.to_string()))?;
                    json!(to_rfc3339(&zoned.to_string()))
                } else {
                    return Err(NtcError::Normalization(format!(
                        "unparseable absolute datetime `{s}`"
                    )));
                }
            }
            SemanticValue::RelativeDate {
                relation,
                weekday,
                offset,
            } => {
                let d = resolve_relative_date(now, tz, *relation, *weekday, offset.as_ref())?;
                json!(d.to_string())
            }
            SemanticValue::RelativeDateTime {
                relation,
                weekday,
                daypart,
                time,
                offset,
            } => {
                let t =
                    time.map(|t| jiff::civil::Time::constant(t.hour as i8, t.minute as i8, 0, 0));
                let s = resolve_relative_datetime(
                    now,
                    tz,
                    policy,
                    *relation,
                    *weekday,
                    *daypart,
                    t,
                    offset.as_ref(),
                )?;
                json!(to_rfc3339(&s))
            }
            SemanticValue::TimeOfDay(t) => json!(format!("{:02}:{:02}", t.hour, t.minute)),
            SemanticValue::Daypart(d) => {
                let t = policy.time_for(*d);
                json!(format!("{:02}:{:02}", t.hour(), t.minute()))
            }
            SemanticValue::List { items, .. } => serde_json::Value::Array(
                items
                    .iter()
                    .map(|item| match item {
                        ntc_core::ir::ListItem::Integer(v) => json!(v),
                        ntc_core::ir::ListItem::Float(v) => json!(v),
                        ntc_core::ir::ListItem::Boolean(v) => json!(v),
                        ntc_core::ir::ListItem::String(v) => json!(v),
                    })
                    .collect(),
            ),
            SemanticValue::Duration(d) => {
                let target = units::target_unit(arg);
                let converted = units::convert(d, target);
                match (arg.param_type, arg.json_type.as_str()) {
                    (_, "integer") => json!(converted.round() as i64),
                    (_, "number") => json!(converted),
                    (ParamType::Duration, "string") => json!(iso8601_duration(d)),
                    _ => json!(converted),
                }
            }
        };
        Ok(v)
    }
}

fn iso8601_duration(d: &ntc_core::ir::DurationValue) -> String {
    use ntc_core::ir::DurationUnit;
    let secs = match d.unit {
        DurationUnit::Second => d.magnitude,
        DurationUnit::Minute => d.magnitude * 60.0,
        DurationUnit::Hour => d.magnitude * 3600.0,
        DurationUnit::Day => d.magnitude * 86_400.0,
        DurationUnit::Week => d.magnitude * 604_800.0,
    }
    .round() as i64;
    let (h, rem) = (secs / 3600, secs % 3600);
    let (m, s) = (rem / 60, rem % 60);
    let mut out = String::from("PT");
    if h > 0 {
        out.push_str(&format!("{h}H"));
    }
    if m > 0 {
        out.push_str(&format!("{m}M"));
    }
    if s > 0 || (h == 0 && m == 0) {
        out.push_str(&format!("{s}S"));
    }
    out
}

impl<B: Backend + ntc_model::AsyncBackend> NeuralToolCompiler<B> {
    /// Async variant of [`Self::compile`] for hosts where GPU readback must
    /// be awaited (wasm/WebGPU).
    pub async fn compile_async(
        &mut self,
        req: &CompileRequest,
    ) -> Result<CompileOutcome, NtcError> {
        let (ids, utterance, inputs) = self.prepare(req)?;
        let outputs = self.backend.run_async(&inputs).await?;
        self.postprocess(req, &ids, &utterance, &inputs, &outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntc_core::ir::*;
    use ntc_model::test_support::{random_weights, test_tokenizer_json, tiny_config};
    use ntc_model::CpuRefBackend;

    fn compiler() -> NeuralToolCompiler<CpuRefBackend> {
        let cfg = tiny_config();
        let tokenizer = NtcTokenizer::from_bytes(test_tokenizer_json().as_bytes()).unwrap();
        let backend = CpuRefBackend::new(cfg.clone(), random_weights(&cfg, 11));
        let mut c = NeuralToolCompiler::from_parts(
            cfg,
            tokenizer,
            backend,
            CompilerConfig {
                timezone: "Europe/Berlin".into(),
                locale: "de-DE".into(),
                ..Default::default()
            },
        );
        c.register_tool(
            serde_json::from_value(serde_json::json!({
                "name": "calendar.create",
                "description": "create a calendar event",
                "parameters": {
                    "title": {"type": "string", "required": true},
                    "start": {"type": "string", "format": "date-time", "required": true},
                    "duration_minutes": {"type": "integer", "semantic": "DURATION"}
                }
            }))
            .unwrap(),
        )
        .unwrap();
        c.register_tool(
            serde_json::from_value(serde_json::json!({
                "name": "email.send",
                "description": "send an email",
                "parameters": {
                    "recipient": {"type": "string", "required": true},
                    "subject": {"type": "string"}
                }
            }))
            .unwrap(),
        )
        .unwrap();
        c
    }

    /// Spec §4 golden: the deterministic backend turns the neural semantic
    /// result into the exact JSON of the spec, independent of the model.
    #[test]
    fn spec_section_4_serialization() {
        let c = compiler();
        let (_, tool) = c.registry.get_by_registry_id("calendar.create").unwrap();
        let ir = ActionIr {
            ir_version: 1,
            action: ActionState::Call,
            action_confidence: 0.998,
            tool: Some(ToolSelection {
                candidate_index: 0,
                registry_id: "calendar.create".into(),
                confidence: 0.997,
            }),
            arguments: vec![
                ArgumentBinding {
                    parameter: "title".into(),
                    value: SemanticValue::String("Zahnarzttermin".into()),
                    confidence: 0.995,
                    provenance: None,
                },
                ArgumentBinding {
                    parameter: "start".into(),
                    value: SemanticValue::RelativeDateTime {
                        relation: DateRelation::Tomorrow,
                        weekday: None,
                        daypart: Some(Daypart::Afternoon),
                        time: None,
                        offset: None,
                    },
                    confidence: 0.981,
                    provenance: None,
                },
                ArgumentBinding {
                    parameter: "duration_minutes".into(),
                    value: SemanticValue::Duration(DurationValue {
                        magnitude: 1.0,
                        unit: DurationUnit::Hour,
                    }),
                    confidence: 0.994,
                    provenance: None,
                },
            ],
            unresolved: vec![],
            ..ActionIr::bare(ActionState::Call, 0.0)
        };
        let tool = tool.clone();
        let req = CompileRequest {
            utterance: "Mach morgen Nachmittag einen einstündigen Zahnarzttermin.".into(),
            locale: None,
            timezone: Some("Europe/Berlin".into()),
            now: Some("2026-08-18T11:00:00+02:00".into()),
            candidates: None,
            context: None,
        };
        let call = c.serialize_call(&ir, &tool, &req).unwrap();
        assert_eq!(call.name, "calendar.create");
        assert_eq!(
            call.arguments,
            serde_json::json!({
                "title": "Zahnarzttermin",
                "start": "2026-08-19T15:00:00+02:00",
                "duration_minutes": 60
            })
        );
    }

    /// Full pipeline smoke: tokenize → pack → CPU forward → decode → policy →
    /// outcome. Random weights make the decision arbitrary but the pipeline
    /// must be deterministic and structurally valid.
    #[test]
    fn compile_end_to_end_deterministic() {
        let mut c = compiler();
        let req = CompileRequest {
            utterance: "make a dentist appointment tomorrow afternoon".into(),
            locale: None,
            timezone: None,
            now: Some("2026-08-18T11:00:00+02:00".into()),
            candidates: None,
            context: None,
        };
        let a = c.compile(&req).unwrap();
        let b = c.compile(&req).unwrap();
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap(),
            "compilation must be deterministic"
        );
        // Whatever the outcome, its IR must be structurally coherent.
        let ir = match &a {
            CompileOutcome::Call { ir, .. }
            | CompileOutcome::Ask { ir, .. }
            | CompileOutcome::NoCall { ir }
            | CompileOutcome::Delegate { ir, .. } => ir,
        };
        assert_eq!(ir.ir_version, 1);
        match ir.action {
            ActionState::Call => assert!(ir.tool.is_some() && ir.unresolved.is_empty()),
            ActionState::Ask => assert!(!ir.unresolved.is_empty()),
            ActionState::NoCall | ActionState::Delegate => {
                assert!(ir.tool.is_none() && ir.arguments.is_empty())
            }
        }
    }

    /// A `LIST<INTEGER>` argument compiles from ONE span: the model marks the
    /// list region, deterministic code splits and parses it (spec §6.2).
    #[test]
    fn list_argument_compiles_from_one_span() {
        let mut c = compiler();
        c.register_tool(
            serde_json::from_value(serde_json::json!({
                "name": "get_data_object",
                "description": "get data objects by id",
                "parameters": {
                    "ids": {"type": "array", "items": {"type": "integer"}, "required": true}
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let (_, tool) = c.registry.get_by_registry_id("get_data_object").unwrap();
        let tool = tool.clone();
        assert_eq!(tool.args[0].param_type, ntc_core::schema::ParamType::List);
        assert_eq!(
            tool.args[0].item_type,
            Some(ntc_core::schema::ParamType::Integer)
        );

        let ir = ActionIr {
            ir_version: 1,
            action: ActionState::Call,
            action_confidence: 0.99,
            tool: Some(ToolSelection {
                candidate_index: 0,
                registry_id: "get_data_object".into(),
                confidence: 0.99,
            }),
            arguments: vec![ArgumentBinding {
                parameter: "ids".into(),
                value: SemanticValue::List {
                    item_type: ntc_core::ir::ListItemType::Integer,
                    items: crate::normalize::list::parse_list(
                        "42, 55 and 101",
                        ntc_core::schema::ParamType::Integer,
                    ),
                    element_provenance: vec![],
                },
                confidence: 0.95,
                provenance: None,
            }],
            unresolved: vec![],
            ..ActionIr::bare(ActionState::Call, 0.0)
        };
        assert!(validate(&ir, Some(&tool)).is_valid());

        let req = CompileRequest {
            utterance: "show data objects 42, 55 and 101".into(),
            locale: None,
            timezone: None,
            now: Some("2026-08-19T10:00:00+02:00".into()),
            candidates: None,
            context: None,
        };
        let call = c.serialize_call(&ir, &tool, &req).unwrap();
        assert_eq!(call.arguments, serde_json::json!({"ids": [42, 55, 101]}));
    }

    /// A tool whose required argument is OPAQUE (free-form object / list of
    /// objects) cannot be compiled into one typed call — policy routes the
    /// utterance to an LLM agent instead of guessing.
    #[test]
    fn opaque_required_argument_routes_to_delegate() {
        use crate::policy::ConfidencePolicy;

        let mut c = compiler();
        c.register_tool(
            serde_json::from_value(serde_json::json!({
                "name": "apply_transition",
                "description": "apply a workflow transition",
                "parameters": {
                    "elements": {"type": "array", "items": {"type": "object"}, "required": true},
                    "workflowName": {"type": "string", "required": true}
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let (_, tool) = c.registry.get_by_registry_id("apply_transition").unwrap();
        let tool = tool.clone();
        assert_eq!(tool.args[0].param_type, ntc_core::schema::ParamType::Opaque);
        assert!(tool.requires_agent());

        let mut ir = ActionIr {
            ir_version: 1,
            action: ActionState::Call,
            action_confidence: 0.96,
            tool: Some(ToolSelection {
                candidate_index: 0,
                registry_id: "apply_transition".into(),
                confidence: 0.96,
            }),
            arguments: vec![ArgumentBinding {
                parameter: "workflowName".into(),
                value: SemanticValue::String("product_review".into()),
                confidence: 0.9,
                provenance: None,
            }],
            unresolved: vec![],
            ..ActionIr::bare(ActionState::Call, 0.0)
        };
        ConfidencePolicy::default().apply(&mut ir, Some(&tool));
        assert_eq!(ir.action, ActionState::Delegate);
        assert!(ir.tool.is_none() && ir.arguments.is_empty());
    }

    /// DELEGATE passes through policy untouched and surfaces the utterance
    /// plus the candidate set, so the host can hand both to an LLM agent.
    #[test]
    fn delegate_outcome_carries_utterance_and_candidates() {
        use crate::policy::ConfidencePolicy;

        let c = compiler();
        let (_, tool) = c.registry.get_by_registry_id("calendar.create").unwrap();
        let tool = tool.clone();
        let mut ir = ActionIr {
            ir_version: 1,
            action: ActionState::Delegate,
            action_confidence: 0.97,
            // A model could still emit a tool guess; policy must clear it.
            tool: Some(ToolSelection {
                candidate_index: 0,
                registry_id: "calendar.create".into(),
                confidence: 0.4,
            }),
            arguments: vec![ArgumentBinding {
                parameter: "title".into(),
                value: SemanticValue::String("x".into()),
                confidence: 0.5,
                provenance: None,
            }],
            unresolved: vec![],
            ..ActionIr::bare(ActionState::Call, 0.0)
        };
        ConfidencePolicy::default().apply(&mut ir, Some(&tool));
        assert_eq!(ir.action, ActionState::Delegate);
        assert!(ir.tool.is_none() && ir.arguments.is_empty() && ir.unresolved.is_empty());

        // Validation accepts a bare DELEGATE and rejects a decorated one.
        assert!(validate(&ir, Some(&tool)).is_valid());
        let mut bad = ir.clone();
        bad.tool = Some(ToolSelection {
            candidate_index: 0,
            registry_id: "calendar.create".into(),
            confidence: 0.9,
        });
        assert!(!validate(&bad, Some(&tool)).is_valid());
    }

    #[test]
    fn explicit_candidates_and_unknown_tool_error() {
        let mut c = compiler();
        let req = CompileRequest {
            utterance: "send an email to the dentist".into(),
            locale: None,
            timezone: None,
            now: Some("2026-08-18T11:00:00+02:00".into()),
            candidates: Some(vec!["email.send".into()]),
            context: None,
        };
        c.compile(&req).unwrap();

        let bad = CompileRequest {
            candidates: Some(vec!["missing.tool".into()]),
            ..req
        };
        assert!(matches!(c.compile(&bad), Err(NtcError::UnknownTool(_))));
    }

    /// A host that registers more tools than fit in one pass used to get
    /// `CandidateLimit` and had to narrow the set itself — which just moved
    /// the routing problem into the host. It now shortlists and decides.
    #[test]
    fn wide_tool_set_shortlists_instead_of_erroring() {
        let mut c = compiler();
        let width = c.slate_limit();
        for i in 0..width * 3 {
            c.register_tool(
                serde_json::from_value(serde_json::json!({
                    "name": format!("filler.tool_{i}"),
                    "description": "an unrelated tool",
                    "parameters": {"q": {"type": "string", "required": true}}
                }))
                .unwrap(),
            )
            .unwrap();
        }
        assert!(c.registry().len() > width, "test needs an oversized set");

        let req = CompileRequest {
            utterance: "send an email to the dentist".into(),
            locale: None,
            timezone: None,
            now: Some("2026-08-18T11:00:00+02:00".into()),
            candidates: None, // the whole registry, as an MCP host would offer
            context: None,
        };
        // Previously an error; now a decision.
        c.compile(&req).unwrap();

        let ids: Vec<ToolId> = c.registry().iter().map(|t| t.id.clone()).collect();
        let s = c.shortlist(&req, &ids).unwrap();
        assert_eq!(s.considered, ids.len());
        assert_eq!(s.kept.len(), width, "shortlist fills exactly one slate");
        assert_eq!(s.scores.len(), ids.len(), "every tool is scored");
        assert_eq!(
            s.rounds,
            ids.len().div_ceil(width),
            "one round per slate, no tool skipped"
        );
        // Scores are sorted, and `kept` is their prefix.
        assert!(s.scores.windows(2).all(|w| w[0].1 >= w[1].1));
        assert_eq!(
            s.kept,
            s.scores[..width]
                .iter()
                .map(|(i, _)| i.clone())
                .collect::<Vec<_>>()
        );
    }

    /// The narrowing must be a pure function of the request: same input, same
    /// slate. Random weights make the *choice* arbitrary, which is precisely
    /// why the property worth pinning is determinism, not which tool wins.
    #[test]
    fn shortlist_is_deterministic() {
        let mut c = compiler();
        for i in 0..8 {
            c.register_tool(
                serde_json::from_value(serde_json::json!({
                    "name": format!("filler.tool_{i}"),
                    "description": "an unrelated tool",
                    "parameters": {"q": {"type": "string"}}
                }))
                .unwrap(),
            )
            .unwrap();
        }
        let req = CompileRequest {
            utterance: "book me a dentist appointment tomorrow afternoon".into(),
            locale: None,
            timezone: None,
            now: Some("2026-08-18T11:00:00+02:00".into()),
            candidates: None,
            context: None,
        };
        let ids: Vec<ToolId> = c.registry().iter().map(|t| t.id.clone()).collect();
        let a = c.shortlist(&req, &ids).unwrap();
        let b = c.shortlist(&req, &ids).unwrap();
        assert_eq!(a.kept, b.kept);
    }
}
