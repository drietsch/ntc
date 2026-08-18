//! Confidence policy (spec §46): threshold-driven downgrades applied to the
//! decoded IR before validation and serialization.

use ntc_core::ir::{ActionIr, ActionState, UnresolvedField, UnresolvedReason};
use ntc_core::schema::CanonicalTool;

#[derive(Debug, Clone, PartialEq)]
pub struct ConfidencePolicy {
    /// Below this tool confidence a CALL becomes NO_CALL.
    pub tool_threshold: f32,
    /// Below this binding confidence a required argument becomes unresolved
    /// (driving ASK).
    pub required_arg_threshold: f32,
}

impl Default for ConfidencePolicy {
    fn default() -> Self {
        Self {
            tool_threshold: 0.35,
            required_arg_threshold: 0.30,
        }
    }
}

impl ConfidencePolicy {
    /// Apply spec §46 downgrades in place. Rules, in order:
    /// 1. CALL with tool confidence below threshold → NO_CALL.
    /// 2. Low-confidence required bindings move to `unresolved` (MISSING).
    /// 3. CALL with any unresolved required field → ASK.
    /// 4. ASK with nothing unresolved → NO_CALL (a model inconsistency;
    ///    fail closed rather than asking an empty question).
    /// 5. NO_CALL drops tool/arguments.
    pub fn apply(&self, ir: &mut ActionIr, tool: Option<&CanonicalTool>) {
        if ir.action == ActionState::Call {
            match (&ir.tool, tool) {
                (Some(sel), Some(_)) if sel.confidence < self.tool_threshold => {
                    ir.action = ActionState::NoCall;
                }
                (None, _) | (_, None) => {
                    ir.action = ActionState::NoCall;
                }
                _ => {}
            }
        }

        if let (ActionState::Call, Some(tool)) = (ir.action, tool) {
            let mut demoted: Vec<UnresolvedField> = vec![];
            ir.arguments.retain(|b| {
                let required = tool.arg(&b.parameter).map(|a| a.required).unwrap_or(false);
                if required && b.confidence < self.required_arg_threshold {
                    demoted.push(UnresolvedField {
                        parameter: b.parameter.clone(),
                        reason: UnresolvedReason::Ambiguous,
                        confidence: b.confidence,
                    });
                    false
                } else {
                    true
                }
            });
            ir.unresolved.extend(demoted);

            if !ir.unresolved.is_empty() {
                ir.action = ActionState::Ask;
            }
        }

        if ir.action == ActionState::Ask && ir.unresolved.is_empty() {
            ir.action = ActionState::NoCall;
        }

        if ir.action == ActionState::NoCall {
            ir.tool = None;
            ir.arguments.clear();
            ir.unresolved.clear();
        }
        if ir.action == ActionState::Ask {
            // ASK keeps the tool selection (the question is about this tool)
            // but arguments are not executable.
            ir.arguments.clear();
        }
    }
}
