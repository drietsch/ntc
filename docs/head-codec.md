# Head Codec v1 (prose companion)

Machine-readable truth: `contracts/heads/v1/head-spec.json`. This page
explains intent; the JSON defines the ABI. Class orderings are FROZEN per
version — any change bumps `head_spec_version` (see contracts/VERSIONS.md).

## Why a head codec exists

The Typed Action IR is the *post-decode* contract. The head codec is the
*neural output ABI*: exact tensor names, shapes, class index assignments,
masking rules, and logit→IR decode rules. It is the single most likely place
for silent training/serving skew, so it is jointly owned and pinned by
tiny-model decision-parity tests (100% agreement gate).

## Decode order (implemented in crates/ntc-runtime/src/decode.rs)

1. `action.logits` argmax → CALL/ASK/NO_CALL (+ temperature-scaled softmax
   confidence).
2. `tool.logits` argmax over n_candidates + NO_TOOL; NO_TOOL ⇒ CALL degrades
   to NO_CALL.
3. Per declared arg of the selected tool: `presence.logits` argmax;
   PRESENT ⇒ decode a value by the arg's canonical TYPE:
   TEXT/PERSON/LOCATION → span; ENUM → enum pointer (index into the schema's
   enum-value list); BOOLEAN → boolean head; INTEGER/FLOAT → span text parsed
   deterministically, magnitude regression as fallback; DURATION → unit head
   + span-parsed magnitude; DATE/DATETIME → factored datetime heads, ABSOLUTE
   relation delegates to span parsing.
4. MISSING/AMBIGUOUS on required args → `unresolved` → policy turns CALL into
   ASK.

## Span rule

start = argmax(start logits over real tokens); end = argmax over
(start, start+32] window, exclusive; resolved to text via tokenizer byte
offsets, skipping zero-width special tokens.

## Anchors

Heads read fused states at anchor positions discovered by byte offset against
the rendered canonical text (crates/ntc-model/src/inputs.rs): the tool's
first token, each `ARG k name`'s name token, each `ENUM j value`'s value
token. The PyTorch model and the Rust runtime share this rule exactly.
