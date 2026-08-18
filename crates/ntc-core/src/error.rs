use thiserror::Error;

/// Spec §64 error taxonomy codes used across validation, eval, and runtime
/// reporting. V1 uses the single-call subset; multi-tool codes are reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TaxonomyCode {
    E01WrongActionState,
    E02WrongTool,
    E03MissingRequiredArgument,
    E04HallucinatedArgument,
    E05WrongArgumentValue,
    E06WrongArgumentBinding,
    E07WrongType,
    E08UnresolvedReference,
    E09FalseReferenceResolution,
    E10UnnecessaryCall,
    E11ShouldHaveAsked,
    E12AskedUnnecessarily,
    E15LocaleNormalizationError,
    E16UnsafeExecutionDecision,
    E17SchemaGeneralizationFailure,
    E18MultilingualSemanticFailure,
}

#[derive(Debug, Error)]
pub enum NtcError {
    #[error("schema error: {0}")]
    Schema(String),

    #[error("unknown tool id: {0}")]
    UnknownTool(String),

    #[error("tool registry full or candidate limit exceeded: {0}")]
    CandidateLimit(String),

    #[error("validation failed: {0:?}")]
    Validation(Vec<crate::validation::ValidationIssue>),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("IR error: {0}")]
    Ir(String),

    #[error("model format error: {0}")]
    Format(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("normalization error: {0}")]
    Normalization(String),
}
