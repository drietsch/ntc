//! NTC-Web core: the deterministic, contract-bearing layer.
//!
//! This crate is the **source of truth** for the cross-workstream contracts:
//! - the Typed Action IR v1 ([`ir`]) — the boundary between learned semantics
//!   and deterministic execution,
//! - the canonical Tool ABI and schema compiler ([`schema`]) — the *only*
//!   implementation of the canonical neural schema text rendering,
//! - IR validation with the spec §64 error taxonomy ([`validation`]),
//! - the tool registry ([`registry`]),
//! - the tokenizer wrapper with token↔byte offset maps ([`tokenizer`]).
//!
//! It has no GPU, wasm-bindgen, or model dependencies and compiles for both
//! native targets and `wasm32-unknown-unknown`.

pub mod error;
pub mod ir;
pub mod registry;
pub mod schema;
pub mod tokenizer;
pub mod validation;

pub use error::NtcError;
pub use ir::{ActionIr, ActionState, ArgumentBinding, CompileRequest, SemanticValue};
pub use registry::{ToolId, ToolRegistry};
pub use schema::{
    CanonicalArg, CanonicalTool, ParamType, RawToolSchema, RiskClass, SemanticTypeId,
};

/// Version of the canonical Tool ABI text rendering. Bump on any change to
/// `CanonicalTool::to_neural_text` output. Recorded in `.ntc` metadata and
/// `contracts/VERSIONS.md`.
///
/// v2 added composite value types (spec §19): `TYPE LIST` with an `ITEM
/// <TYPE>` line, `TYPE OPAQUE` for agent-only payloads, and flattened
/// `parent.child` arguments for objects with declared scalar properties.
pub const ABI_VERSION: u32 = 2;

/// Version of the Typed Action IR emitted and accepted by this crate.
pub const IR_VERSION: u32 = 1;
