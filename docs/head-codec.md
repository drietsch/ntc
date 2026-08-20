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
   PRESENT ⇒ decode a value, in this order:
   a. `filter_template.logits`, if the arg carries a `SEMANTIC` some declared
      template serves — see below;
   b. `source.logits`: LINKED_ITEM → entity-ref head; RESOLVER → the host's
      pre-pass; MODEL → the schema's declared `default`, if it states one;
   c. otherwise by the arg's canonical TYPE: TEXT/PERSON/LOCATION → span;
      ENUM → enum pointer (index into the schema's enum-value list);
      BOOLEAN → boolean head; INTEGER/FLOAT → span text parsed
      deterministically; DURATION → unit head + span-parsed magnitude;
      DATE/DATETIME → factored datetime heads, ABSOLUTE relation delegates to
      span parsing.
4. MISSING/AMBIGUOUS on required args → `unresolved` → policy turns CALL into
   ASK.

## Values the request does not contain (v4)

Some arguments want a value that appears nowhere in the utterance: a PQL
filter, a query expression. "which BrandAsset have a photographer below 10?"
asks for `photographer < 10`, and no span produces that string.

The tempting conclusion is that such arguments are beyond an architecture with
no decoder, and this project drew it — reporting an "88.8% ceiling" on the
Studio corpus. That was mostly wrong. Those values are not free-form: they
take a handful of shapes the host knows, and everything that varies inside a
shape *is* in the request. So the compiler does not have to write the value,
only to choose its shape and fill the blanks.

`filter_template` is that choice. Class 0 is NONE; class *i+1* is
`model.filter_templates[i]`, and the logits are masked to NONE plus the
templates whose `semantic` matches the argument's own `SEMANTIC` annotation —
the same masking the enum head does against an argument's enum values. The
winning pattern is then rendered by deterministic code from the *same span the
span head already marks*: `{field}` takes the first identifier-shaped word in
it, `{number}` the last number, `{token}` the declared value it names.

This is the datetime head's trick, not a new idea: pick `NEXT` + `FRIDAY` and
let deterministic code produce the date.

Two properties matter more than the accuracy it buys:

- **A slot the span cannot fill produces no value at all.** The argument stays
  unresolved and the call becomes ASK. A confidently wrong filter executes
  silently against the customer's data; a question does not.
- **The vocabulary is host data, not contract.** The table lives in `.ntc`
  metadata because the head's class order is fixed by what the model trained
  against. A host adds a template by retraining, not by a `head_spec_version`
  bump.

The related trap is worth naming: NONE has to be supervised by rows where an
argument a template *could* serve is simply not filled, or "servable" collapses
into "templated". The Studio corpus has 432 such rows against 258 positives.
Wide-slate routing learned this the hard way — NO_TOOL was trained only on
"nothing was asked of me" and so never meant "none of these fits".

## Span rule

start = argmax(start logits over real tokens); end = argmax over
(start, start+32] window, exclusive; resolved to text via tokenizer byte
offsets, skipping zero-width special tokens.

## Anchors

Heads read fused states at anchor positions discovered by byte offset against
the rendered canonical text (crates/ntc-model/src/inputs.rs): the tool's
first token, each `ARG k name`'s name token, each `ENUM j value`'s value
token. The PyTorch model and the Rust runtime share this rule exactly.
