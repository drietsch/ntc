//! Tool registry and V1 candidate selection (spec §21, trimmed: candidates
//! are caller-supplied or "all registered tools" when ≤ MAX_CANDIDATES).

use std::collections::HashMap;

use crate::error::NtcError;
use crate::schema::{compile_schema, CanonicalTool, RawToolSchema};

/// V1 candidate limit (spec §73).
pub const MAX_CANDIDATES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ToolId(pub u32);

#[derive(Debug)]
pub struct RegisteredTool {
    pub id: ToolId,
    pub canonical: CanonicalTool,
}

/// Holds compiled canonical tools keyed by registry id.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: Vec<RegisteredTool>,
    by_registry_id: HashMap<String, ToolId>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compile and register a raw schema. Re-registering an existing registry
    /// id replaces the previous definition (schema update).
    pub fn register(&mut self, raw: RawToolSchema) -> Result<ToolId, NtcError> {
        let canonical = compile_schema(&raw)?;
        if let Some(&existing) = self.by_registry_id.get(&canonical.id) {
            self.tools[existing.0 as usize].canonical = canonical;
            return Ok(existing);
        }
        let id = ToolId(self.tools.len() as u32);
        self.by_registry_id.insert(canonical.id.clone(), id);
        self.tools.push(RegisteredTool { id, canonical });
        Ok(id)
    }

    pub fn get(&self, id: ToolId) -> Option<&CanonicalTool> {
        self.tools.get(id.0 as usize).map(|t| &t.canonical)
    }

    pub fn get_by_registry_id(&self, registry_id: &str) -> Option<(ToolId, &CanonicalTool)> {
        self.by_registry_id
            .get(registry_id)
            .map(|&id| (id, &self.tools[id.0 as usize].canonical))
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &RegisteredTool> {
        self.tools.iter()
    }

    /// Resolve the candidate set for a compile request (V1 semantics):
    /// explicit registry ids in candidate order, or all registered tools if
    /// `None` and the registry holds ≤ [`MAX_CANDIDATES`] tools.
    pub fn resolve_candidates(&self, explicit: Option<&[String]>) -> Result<Vec<ToolId>, NtcError> {
        let ids: Vec<ToolId> = match explicit {
            Some(ids) => ids
                .iter()
                .map(|rid| {
                    self.by_registry_id
                        .get(rid)
                        .copied()
                        .ok_or_else(|| NtcError::UnknownTool(rid.clone()))
                })
                .collect::<Result<_, _>>()?,
            None => self.tools.iter().map(|t| t.id).collect(),
        };
        if ids.is_empty() {
            return Err(NtcError::CandidateLimit("no candidate tools".into()));
        }
        if ids.len() > MAX_CANDIDATES {
            return Err(NtcError::CandidateLimit(format!(
                "{} candidates exceeds the V1 limit of {MAX_CANDIDATES}; pass an explicit candidate list",
                ids.len()
            )));
        }
        Ok(ids)
    }
}

/// Seam for the post-V1 retriever (spec §21–22). The V1 implementation is the
/// registry's own `resolve_candidates`.
pub trait CandidateSelector {
    fn select(
        &self,
        utterance: &str,
        registry: &ToolRegistry,
        k: usize,
    ) -> Result<Vec<ToolId>, NtcError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(name: &str) -> RawToolSchema {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "description": "d",
            "parameters": {}
        }))
        .unwrap()
    }

    #[test]
    fn register_and_resolve() {
        let mut reg = ToolRegistry::new();
        let a = reg.register(raw("a.x")).unwrap();
        let b = reg.register(raw("b.y")).unwrap();
        assert_ne!(a, b);
        // Re-register replaces, keeps id.
        let a2 = reg.register(raw("a.x")).unwrap();
        assert_eq!(a, a2);
        assert_eq!(reg.len(), 2);

        let all = reg.resolve_candidates(None).unwrap();
        assert_eq!(all, vec![a, b]);
        let explicit = reg.resolve_candidates(Some(&["b.y".to_string()])).unwrap();
        assert_eq!(explicit, vec![b]);
        assert!(reg
            .resolve_candidates(Some(&["missing".to_string()]))
            .is_err());
    }

    #[test]
    fn candidate_limit_enforced() {
        let mut reg = ToolRegistry::new();
        for i in 0..(MAX_CANDIDATES + 1) {
            reg.register(raw(&format!("t.{i}"))).unwrap();
        }
        assert!(reg.resolve_candidates(None).is_err());
    }
}
