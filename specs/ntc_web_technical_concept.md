# NTC-Web
## Technical Concept: A Multilingual Neural Compiler for Typed Tool Calling on WebGPU

**Status:** Technical concept / research architecture  
**Version:** 0.2  
**Date:** 2026-08-18  
**Working name:** NTC-Web  
**Primary implementation target:** Rust + WebAssembly + WebGPU  
**Preferred architecture:** Multilingual encoder + schema encoder + fusion + structured prediction heads  
**Target weight precision:** 1.58-bit ternary `{-1, 0, +1}`  
**Primary objective:** Maximize executable semantic correctness per byte, joule, and millisecond.

---

# 1. Executive Summary

NTC-Web is a purpose-built **multilingual neural compiler** that translates human intent into typed, executable tool calls.

The system is not intended to be a general-purpose language model. It is designed from first principles around one task:

> **Understand human intent, understand available tool schemas, and compile the two into a typed executable action.**

The core principle is:

> **Do not compress an LLM until it becomes a tool caller. Design a neural tool-call compiler from first principles, and only retain the pieces of an LLM architecture that help that task.**

NTC-Web therefore rejects the standard architecture of:

```text
user prompt
  -> decoder-only LLM
  -> autoregressive JSON
```

and instead uses:

```text
user language
  -> multilingual semantic encoder
  -> schema encoder
  -> schema-language fusion
  -> structured prediction heads
  -> Typed Action IR
  -> deterministic Rust compiler
  -> validated executable tool call
```

The neural component is responsible only for uncertain semantic interpretation:

- multilingual language understanding,
- intent recognition,
- tool selection,
- argument mapping,
- entity extraction,
- reference resolution,
- ambiguity detection,
- multi-tool dependency planning,
- confidence estimation.

Deterministic Rust/WASM code is responsible for:

- JSON serialization,
- schema validation,
- enum validity,
- type enforcement,
- date/time normalization,
- unit conversion,
- policy and permission checks,
- execution orchestration.

All neural inference is designed to run locally in the browser through **WebGPU compute shaders**, orchestrated by Rust compiled to WebAssembly.

The initial research target is **~250M parameters**, with a model ladder from roughly 125M to 400M or beyond. The end-state compression target is 1.58-bit ternary weights with low-bit activations.

The system should be evaluated as a **semantic compiler**, not a chatbot.

---

# 2. Core Thesis

The central research question is not:

> How small can an LLM be?

It is:

> **What is the minimum learned semantic capacity required to generalize from multilingual human intent to previously unseen typed tool schemas?**

Everything in NTC-Web follows from this question.

The design deliberately avoids spending parameters and inference time on:

- prose generation,
- broad world knowledge,
- generic coding,
- open-ended conversation,
- storytelling,
- reasoning traces,
- JSON punctuation,
- syntax generation,
- unconstrained token continuation.

The intended capability frontier is:

```text
smallest possible learned semantic engine
+
deterministic compiler backend
+
browser-local execution
```

---

# 3. Product Definition

NTC-Web receives:

1. a user utterance,
2. optional conversational context,
3. a set of candidate tool schemas,
4. optional environment metadata,
5. optional previous tool results,

and returns one of:

```text
CALL
CALL_SEQUENCE
ASK
NO_CALL
REQUEST_APPROVAL
DENY
ERROR
```

The output is a **typed action program**, not natural language.

---

# 4. Example

## User

```text
Mach morgen Nachmittag einen einstündigen Zahnarzttermin.
```

## Neural semantic result

```yaml
action: CALL

tool:
  index: 2
  id: calendar.create

arguments:
  title:
    semantic_type: STRING
    value: "Zahnarzttermin"

  start:
    semantic_type: RELATIVE_DATETIME
    relation: TOMORROW
    daypart: AFTERNOON

  duration:
    semantic_type: DURATION
    magnitude: 1
    unit: HOUR
```

## Deterministic Rust backend

```json
{
  "name": "calendar.create",
  "arguments": {
    "title": "Zahnarzttermin",
    "start": "2026-08-19T15:00:00+02:00",
    "duration_minutes": 60
  }
}
```

The neural model does not generate the braces, commas, field ordering, ISO timestamp formatting, or schema validation logic.

---

# 5. Explicit Non-Goals

NTC-Web is deliberately not optimized for:

- general question answering,
- conversational personality,
- long-form prose,
- creative writing,
- coding assistance,
- document generation,
- factual encyclopedic recall,
- general reasoning benchmarks,
- unrestricted text generation,
- image understanding,
- speech generation,
- broad agentic behavior.

If some of these capabilities appear incidentally, they are not primary optimization targets.

---

# 6. Design Principles

## 6.1 Neural where meaning is uncertain

Use learned components for:

```text
intent
tool relevance
argument binding
entity semantics
reference resolution
ambiguity
cross-lingual understanding
dependency planning
confidence
```

## 6.2 Deterministic where rules are exact

Use Rust/WASM for:

```text
JSON
types
schemas
enums
dates
units
permissions
execution
validation
serialization
retries
```

## 6.3 Structured prediction over free-form generation

Prefer:

```text
classify
point
select
bind
copy
normalize
```

over:

```text
generate arbitrary tokens
```

## 6.4 Browser inference is a first-class architectural constraint

The neural architecture must map efficiently onto portable WebGPU primitives.

Model architecture and browser inference architecture are designed together, not independently.

## 6.5 No autoregressive JSON

The system should not invoke the full network repeatedly just to emit structural syntax.

---

# 7. Preferred Research Architecture

NTC-Web uses **Architecture B**:

```text
Multilingual semantic encoder
        +
Schema encoder
        +
Schema-language fusion
        +
Structured prediction heads
        +
Optional tiny string decoder
```

A decoder-only Transformer is retained only as a baseline for comparison.

---

# 8. High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        Browser Runtime                       │
│                                                              │
│                  Rust compiled to WASM                      │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Tokenizer / Context │
                 │ Rust / WASM CPU     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Tool Retriever      │
                 │ Rust / WASM / GPU   │
                 └──────────┬──────────┘
                            │
                          top-K
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Schema Compiler     │
                 │ Rust / WASM CPU     │
                 └──────────┬──────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                         WebGPU                               │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Multilingual Semantic Encoder                         │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼─────────────────────────────────┐  │
│  │ Schema Encoder / Candidate Tool Encoder               │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼─────────────────────────────────┐  │
│  │ Schema-Language Fusion                               │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                         │                                    │
│        ┌────────────────┼────────────────┐                   │
│        ▼                ▼                ▼                   │
│   Action Head       Tool Head      Argument Heads            │
│        │                │                │                   │
│        └────────────────┼────────────────┘                   │
│                         ▼                                    │
│                  Typed predictions                           │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Typed Action IR     │
                 │ Rust / WASM CPU     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Validation / Policy │
                 │ Rust / WASM CPU     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ JSON / API Adapter  │
                 │ Rust / WASM CPU     │
                 └──────────┬──────────┘
                            │
                            ▼
                       TOOL EXECUTION
```

---

# 9. Why This Architecture Fits WebGPU

A conventional decoder model executes the network once per generated token.

For a 30-token function call:

```text
30 generated tokens
≈ 30 repeated model passes
```

NTC-Web aims to produce most of the action in one or a few structured passes:

```text
encode utterance
encode schemas
fuse
predict action/tool/arguments
done
```

Benefits:

- fewer GPU dispatches,
- no ordinary autoregressive loop,
- no KV cache for simple calls,
- no sampling,
- no temperature,
- no beam search,
- no repetition penalties,
- no JSON token generation,
- lower latency,
- lower memory traffic,
- simpler browser runtime.

---

# 10. Decoder Strategy

## 10.1 V1 recommendation: no general decoder

The first version should use:

```text
encoder
+
fusion
+
prediction heads
```

with no general sequence decoder.

Only genuinely free-form fields may use a tiny optional micro-decoder.

## 10.2 Why

Most tool-call fields can be expressed as:

- classification,
- span extraction,
- pointer selection,
- numeric semantics,
- enum selection,
- entity reference,
- structured time/date representation.

This makes arbitrary autoregressive generation unnecessary for the majority of calls.

---

# 11. Suggested Parameter Budget

Initial target:

```text
NTC-Web-250M
```

Possible budget:

```text
Multilingual encoder            160M
Schema encoder                   30M
Fusion layers                    30M
Structured output heads          15M
Optional string micro-decoder    15M
------------------------------------
Total                           250M
```

Research ladder:

```text
NTC-Web-125M
NTC-Web-250M
NTC-Web-400M
NTC-Web-600M
```

The objective is to identify the smallest model that preserves strong unseen-schema generalization.

---

# 12. Input Representation

Input consists of:

```yaml
utterance: string

context:
  prior_user_turns: optional
  prior_tool_results: optional
  entities: optional
  locale: optional
  timezone: optional
  units: optional
  currency: optional

candidate_tools:
  - canonical tool schema
```

The browser should not insert thousands of raw tools into the neural context.

---

# 13. Multilingual Semantic Encoder

The encoder must learn a shared semantic space across languages.

Examples:

```text
Turn off the bedroom light.
Mach das Licht im Schlafzimmer aus.
Éteins la lumière de la chambre.
Apaga la luz del dormitorio.
寝室の照明を消して。
```

should map to closely aligned intent representations.

The encoder should be optimized for:

- intent equivalence,
- entity extraction,
- relation extraction,
- quantities,
- dates and times,
- references,
- negation,
- modality,
- request vs mention distinction,
- code switching,
- colloquial phrasing.

---

# 14. Schema Encoder

The schema encoder receives **canonical compiled schemas**, not arbitrary verbose provider JSON whenever possible.

Raw tool schemas should first be normalized.

Example raw schema:

```json
{
  "name": "search_trains",
  "description": "Search for available train journeys",
  "parameters": {
    "destination_city": {
      "type": "string"
    },
    "departure_time": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

Canonical form:

```text
TOOL 7
DESC search available train journeys

ARG 0 destination_city
TYPE TEXT
REQUIRED 1
SEMANTIC LOCATION.DESTINATION

ARG 1 departure_time
TYPE DATETIME
REQUIRED 1
```

The canonical format should be compact, stable, versioned, and independent of external schema provider formatting.

---

# 15. Schema-Language Fusion

The fusion stage resolves user semantics against candidate schema semantics.

Possible implementations:

```text
cross-attention
late fusion
shared latent slots
candidate-conditioned attention
```

A practical initial design:

```text
user embedding sequence
        +
tool/argument embedding sequence
        ↓
cross-attention fusion
        ↓
tool-specific semantic states
```

The output should support per-tool and per-argument prediction.

---

# 16. Structured Output Heads

The system should avoid autoregressive generation wherever possible.

## 16.1 Action head

Predict one of:

```text
CALL
CALL_SEQUENCE
ASK
NO_CALL
REQUEST_APPROVAL
DENY
ERROR
```

## 16.2 Tool pointer head

Select one candidate tool index:

```text
P(tool_i | user, candidate_tools)
```

No tool name generation is required.

## 16.3 Argument presence head

For each parameter:

```text
PRESENT
MISSING
AMBIGUOUS
NOT_APPLICABLE
```

## 16.4 Span pointer head

For values present directly in the input:

```text
"Send it to Anna Müller."
            ^^^^^^^^^^^
```

predict the source span.

Useful for:

- names,
- addresses,
- emails,
- filenames,
- IDs,
- URLs,
- copied text.

## 16.5 Enum head

For:

```text
priority = [low, normal, high]
```

predict an enum index.

## 16.6 Numeric semantic head

Example:

```text
"one and a half hours"
```

becomes:

```yaml
magnitude: 1.5
unit: HOUR
```

## 16.7 Entity reference head

Example:

```text
"Send it to her."
```

maps to a known context entity pointer.

## 16.8 Date/time semantic head

Example:

```text
"next Friday afternoon"
```

becomes:

```yaml
relation: NEXT
weekday: FRIDAY
daypart: AFTERNOON
```

Deterministic code resolves the final timestamp.

## 16.9 Dependency head

For multi-tool workflows, predict edges:

```text
step_1.output -> step_2.argument
```

## 16.10 Optional free-string micro-decoder

Only activate for fields requiring actual text generation.

Examples:

```text
email body
message body
search query reformulation
```

This decoder should remain small and optional.

---

# 17. Typed Action IR

The Typed Action IR is the boundary between learned semantics and deterministic execution.

It should be:

- versioned,
- compact,
- language-neutral,
- strongly typed,
- schema-aware,
- provenance-aware,
- confidence-aware,
- suitable for single and multi-step calls.

---

# 18. IR Example

```yaml
ir_version: 1

action: CALL

tool:
  candidate_index: 2
  registry_id: calendar.create
  confidence: 0.997

arguments:
  - parameter: title
    semantic_type: STRING
    value: "Zahnarzttermin"
    confidence: 0.995
    provenance:
      source: user
      span: [33, 51]

  - parameter: start
    semantic_type: RELATIVE_DATETIME
    value:
      relation: TOMORROW
      daypart: AFTERNOON
    confidence: 0.981

  - parameter: duration
    semantic_type: DURATION
    value:
      magnitude: 1
      unit: HOUR
    confidence: 0.994

unresolved: []
```

---

# 19. Semantic Types

Recommended built-in semantic types:

```text
STRING
BOOLEAN
INTEGER
FLOAT
ENUM
IDENTIFIER
PERSON_REF
ORGANIZATION_REF
LOCATION
ABSOLUTE_DATE
RELATIVE_DATE
ABSOLUTE_DATETIME
RELATIVE_DATETIME
TIME_OF_DAY
DAYPART
DURATION
MONEY
PERCENTAGE
UNIT_VALUE
PHONE
EMAIL
URI
FILE_REF
EVENT_REF
RESOURCE_REF
LIST<T>
OBJECT<T>
```

These are richer than JSON types and enable deterministic canonicalization.

---

# 20. Multi-Tool IR

Example:

```yaml
action: CALL_SEQUENCE

steps:
  - id: s1
    tool: contacts.search
    arguments:
      query:
        semantic_type: STRING
        value: "Tom"

  - id: s2
    tool: calendar.search
    arguments:
      date:
        semantic_type: RELATIVE_DATE
        relation: TOMORROW
      query:
        semantic_type: STRING
        value: "project review"

  - id: s3
    tool: email.send
    depends_on:
      - s1
      - s2
    arguments:
      recipient:
        from_step: s1
        path: person.email

      body:
        semantic_type: GENERATED_STRING
        template:
          text: "Tomorrow's project review is at {location}."
          bindings:
            location:
              from_step: s2
              path: event.location
```

---

# 21. Tool Retrieval

The model should receive only a small candidate set.

Example:

```text
10,000 registered tools
        ↓
retrieval
        ↓
top 8–32
        ↓
NTC-Web
```

Possible retrieval strategies:

- dense multilingual embeddings,
- sparse lexical retrieval,
- hybrid search,
- hierarchical categories,
- learned bi-encoder,
- graph-aware retrieval.

The retriever optimizes recall.

The neural compiler optimizes precision.

---

# 22. Dependency-Aware Retrieval

The tool registry should maintain semantic compatibility metadata:

```text
tool output type
        ↓
compatible tool input type
```

Example:

```text
contacts.search.person_id
    -> email.send.recipient_id
```

Candidate expansion may therefore use:

```text
initial top-K
+
dependency-compatible neighbors
```

This is useful for multi-step actions.

---

# 23. Runtime Responsibility Split

## Rust/WASM CPU responsibilities

```text
tokenization
text preprocessing
schema parsing
schema canonicalization
tool registry
retrieval orchestration
Typed Action IR construction
date/time normalization
unit conversion
permission checks
policy logic
schema validation
JSON serialization
API invocation
execution state
error handling
```

## WebGPU responsibilities

```text
embeddings
encoder blocks
attention
feed-forward layers
schema encoding
cross-attention/fusion
tool scoring
argument scoring
pointer heads
enum heads
classification heads
dependency scoring
```

The GPU should perform tensor-heavy work.

The CPU/WASM runtime should perform control-heavy deterministic work.

---

# 24. WebAssembly Target

Primary Rust target:

```text
wasm32-unknown-unknown
```

The browser package should contain approximately:

```text
index.html
ntc.js
ntc_bg.wasm
model.ntc
```

JavaScript should remain minimal and used only for browser host interop.

The application and inference logic should remain in Rust.

---

# 25. WebGPU Backend

Rust GPU abstraction:

```text
wgpu
```

Shader language:

```text
WGSL
```

The runtime should compile or load WGSL compute shaders.

Example kernels:

```text
embedding
ternary matmul
attention
softmax
normalization
fusion
tool head
argument head
pointer head
enum head
```

---

# 26. WebGPU Portability Rule

The baseline implementation should target the broadly portable WebGPU feature set.

Do not make correctness dependent on:

- subgroup operations,
- native ternary arithmetic,
- browser-specific extensions,
- native-only integer-width features,
- vendor-specific intrinsics.

Optional fast paths may be selected at runtime when supported.

---

# 27. Weight Precision

Target semantic weights:

```text
{-1, 0, +1}
```

Ideal information density:

```text
log2(3) ≈ 1.585 bits / weight
```

Two runtime formats should be benchmarked.

---

# 28. T2 Weight Format

Simple 2-bit encoding:

```text
00 = 0
01 = +1
10 = -1
11 = reserved
```

One `u32` contains 16 weights.

Advantages:

- simple bit extraction,
- cheap WGSL implementation,
- predictable alignment,
- efficient buffer storage,
- portable.

Disadvantage:

```text
2.0 bits / weight
```

instead of ideal 1.585.

Approximate raw storage:

```text
125M params -> 31.25 MB
250M params -> 62.5 MB
400M params -> 100 MB
```

---

# 29. TritPack20 Weight Format

Because:

```text
3^20 < 2^32
```

20 ternary values can theoretically be represented in one 32-bit word.

Storage density:

```text
32 / 20 = 1.6 bits / weight
```

Approximate raw storage:

```text
125M params -> 25 MB
250M params -> 50 MB
400M params -> 80 MB
```

Advantages:

- near-ideal ternary packing,
- lower bandwidth.

Disadvantages:

- radix-3 unpacking,
- expensive division/modulo or equivalent decoding,
- likely slower kernels.

This format should be treated as an experimental bandwidth optimization.

---

# 30. Kernel Strategy

Benchmark two primary ternary matmul paths:

```text
Kernel A: T2 fast unpack
Kernel B: TritPack20 compact unpack
```

The winner should be determined by actual browser benchmarks.

Theoretical file size alone should not determine the runtime representation.

---

# 31. Activation Precision

The practical long-term target should consider both weights and activations.

Candidate configurations:

```text
W1.58A16
W1.58A8
W1.58A4
```

Initial recommendation:

```text
train/reference: BF16 or FP16
browser baseline: A8 or F16
research target: W1.58A4
```

The best choice depends on accuracy, hardware support, and kernel efficiency.

---

# 32. WebGPU Runtime Capability Selection

At startup:

```text
request adapter
    ↓
inspect limits/features
    ↓
select kernel backend
```

Possible paths:

```text
portable u32 ternary kernels
shader-f16 path
packed int8 dot-product path
future subgroup path
```

The model format should remain independent of a single kernel implementation.

---

# 33. Model File Format

Use a custom runtime format:

```text
.ntc
```

Import/export can support other formats during training, but the browser runtime should load a format optimized for WebGPU.

---

# 34. `.ntc` File Structure

Example:

```text
NTC1
│
├── header
│   ├── format_version
│   ├── architecture
│   ├── model_version
│   ├── quantization
│   ├── tokenizer_version
│   └── tensor_count
│
├── tokenizer
│
├── semantic_type_table
│
├── tensor_directory
│
├── encoder.layer.0
├── encoder.layer.1
├── ...
├── schema_encoder
├── fusion
├── action_head
├── tool_head
├── argument_heads
└── calibration_data
```

---

# 35. Tensor Metadata

Example:

```yaml
name: encoder.layer.3.ffn.weight
dtype: TERNARY_T2
shape:
  - 1536
  - 4096
scale_format: F16
offset: 12345678
length: 3145728
```

Alternative:

```yaml
dtype: TERNARY_PACK20
```

Weights should be uploaded to GPU buffers without expensive full-model conversion.

---

# 36. Buffer Strategy

Do not assume a single giant GPU buffer.

Use per-layer or per-matrix buffers.

Example:

```text
layer00.attn
layer00.ffn
layer01.attn
layer01.ffn
...
```

Benefits:

- easier compatibility with device limits,
- easier streaming,
- simpler profiling,
- reduced initialization complexity,
- easier partial loading.

---

# 37. Memory Strategy

Persistent GPU memory:

```text
packed weights
small scale tensors
embedding tables
model constants
```

Transient GPU memory:

```text
activations
attention buffers
fusion states
head outputs
```

Reuse transient buffers aggressively.

Since the architecture avoids a general autoregressive decoder, ordinary tool calls should not require a large KV cache.

---

# 38. Tokenizer

The tokenizer must prioritize semantic efficiency across languages.

Evaluation criteria:

```text
tokens per utterance
tokens per entity
tokens per schema description
multilingual fragmentation
numeric fragmentation
date/time fragmentation
identifier fragmentation
```

Tokenizer design should avoid excessive fragmentation of:

- names,
- URLs,
- emails,
- IDs,
- dates,
- numbers,
- code-like tool names.

---

# 39. Schema Compiler

Raw schemas may arrive as:

```text
JSON Schema
OpenAPI
custom JSON
internal Rust definitions
```

The schema compiler normalizes them into a canonical ABI.

Pipeline:

```text
raw schema
   ↓
parse
   ↓
normalize
   ↓
semantic annotate
   ↓
canonical neural schema
```

---

# 40. Tool ABI

Each tool should be represented internally as:

```yaml
id: calendar.create
version: 3

description: Create a calendar event

risk:
  class: WRITE
  requires_approval: false

arguments:
  title:
    type: string
    semantic_type: EVENT_TITLE
    required: true

  start:
    type: datetime
    semantic_type: DATETIME
    required: true

  duration_minutes:
    type: integer
    semantic_type: DURATION_MINUTES
    required: false

returns:
  event_id:
    type: resource_ref
    semantic_type: CALENDAR_EVENT
```

---

# 41. ABI Versioning

Pipeline:

```text
External schema
      ↓
Schema Compiler vN
      ↓
Canonical Tool ABI
      ↓
NTC-Web model
      ↓
Action IR vM
      ↓
Backend adapter
```

The model should not depend on provider-specific JSON formatting.

---

# 42. Deterministic Compiler Backend

The compiler backend receives the Typed Action IR.

Responsibilities:

```text
resolve relative dates
normalize units
resolve locale conventions
bind entity IDs
enforce schema types
check required fields
apply permission policy
serialize JSON
invoke API
```

---

# 43. Date/Time Normalization

Example neural output:

```yaml
type: RELATIVE_DATETIME
relation: TOMORROW
daypart: AFTERNOON
```

Rust runtime resolves:

```text
user timezone
current local date
deployment policy for "afternoon"
```

and converts to a canonical datetime.

The model should not be forced to learn timezone arithmetic when deterministic code can do it exactly.

---

# 44. Unit Normalization

Example:

```text
"one and a half hours"
```

Neural:

```yaml
magnitude: 1.5
unit: HOUR
```

Runtime:

```json
{
  "duration_minutes": 90
}
```

---

# 45. Policy and Safety Layer

The neural model must never be the sole security boundary.

Execution pipeline:

```text
Typed Action IR
      ↓
schema validation
      ↓
permission policy
      ↓
risk policy
      ↓
approval requirement
      ↓
execution
```

High-impact actions may require explicit approval.

Examples:

- deleting files,
- sending money,
- publishing content,
- changing credentials,
- sending external messages,
- account changes.

---

# 46. Confidence and Abstention

The model should output calibrated confidence.

Example:

```yaml
action_confidence: 0.998
tool_confidence: 0.992

arguments:
  recipient:
    confidence: 0.61
```

The runtime can enforce thresholds:

```text
low tool confidence      -> ASK / NO_CALL
low critical arg confidence -> ASK
high-risk action         -> REQUEST_APPROVAL
```

---

# 47. Training Philosophy

The student should be trained as a compiler, not a chatbot.

Primary targets:

```text
action classification
tool selection
argument presence
argument binding
semantic value extraction
reference resolution
dependency prediction
confidence calibration
```

Generic next-token perplexity is secondary.

---

# 48. Training Stages

## Stage 0 — Initialization

Benchmark:

```text
existing multilingual encoder
distilled custom encoder
train-from-scratch encoder
```

Recommended first path:

```text
start from a compact multilingual pretrained backbone
```

Then progressively specialize.

---

## Stage 1 — Semantic grounding

Train tasks relevant to tool use:

```text
intent similarity
slot filling
entity extraction
relation extraction
coreference
date/time understanding
quantity understanding
multilingual paraphrase alignment
negation
request vs mention
```

---

## Stage 2 — Schema grounding

Train on tool definitions and requests.

Critical requirement:

```text
unseen schema generalization
```

The model must understand tool descriptions rather than memorize names.

---

# 49. Tool and Argument Name Randomization

Example original:

```text
send_email
recipient
subject
body
```

Randomized:

```text
fn_193
arg_0
arg_1
arg_2
```

Descriptions remain semantically meaningful.

This forces semantic schema reading.

Training variants should include:

```text
masked tool names
masked parameter names
randomized names
reordered arguments
paraphrased descriptions
misleading names
near-duplicate tools
```

---

# 50. Hard Negatives

Examples:

## Mention vs request

```text
"What does delete_account do?"
```

→ `NO_CALL`

```text
"Delete my account."
```

→ `REQUEST_APPROVAL` or `CALL`

## Missing information

```text
"Send the report."
```

→ `ASK(recipient)`

## Unsupported action

No valid tool exists.

→ `NO_CALL`

## Ambiguous entity

Two matching entities exist.

→ `ASK`

---

# 51. Synthetic Data Engine

Use one or more capable teacher models to generate:

```text
tool schemas
user requests
semantic IR
valid calls
hard negatives
ambiguity cases
multilingual variants
code-switched requests
adversarial cases
multi-turn contexts
execution failures
recovery cases
```

---

# 52. Data Mutation Pipeline

For each base example:

```text
paraphrase
translate
code-switch
inject typo
use slang
change word order
omit subject
replace entity with pronoun
add irrelevant detail
introduce ambiguity
change locale
rename tool
rename argument
reorder schema
add decoy tool
remove required field
add conflicting detail
```

---

# 53. Counterfactual Training

Base:

```text
"Email Anna the report."
```

Counterfactuals:

```text
"Show me Anna's email address."
"Draft an email to Anna, but don't send it."
"Did I already email Anna the report?"
"Delete the report Anna sent me."
"Email Anna's report to Ben."
```

This reduces keyword-triggered false calls.

---

# 54. Distillation

Distill decision structure rather than prose reasoning.

Teacher outputs may provide:

```text
action distribution
tool distribution
argument alignment
semantic types
missing-field decisions
dependency graph
confidence
```

Avoid training the student to reproduce verbose chain-of-thought.

---

# 55. Verifier-Based Optimization

Tool calling supports strong automated verification.

Reward dimensions:

```text
correct action
correct tool
correct required args
correct optional args
semantic equivalence
successful execution
minimal call sequence
no hallucinated fields
no unnecessary call
no policy violation
```

The final objective should emphasize semantic execution correctness.

---

# 56. Loss Design

Conceptual objective:

\[
L =
\lambda_a L_{action}
+ \lambda_t L_{tool}
+ \lambda_p L_{presence}
+ \lambda_b L_{binding}
+ \lambda_v L_{value}
+ \lambda_r L_{reference}
+ \lambda_d L_{dependency}
+ \lambda_c L_{calibration}
+ \lambda_{kd} L_{distillation}
\]

Where:

- \(L_{action}\): action state classification,
- \(L_{tool}\): candidate tool selection,
- \(L_{presence}\): field presence/missing/ambiguity,
- \(L_{binding}\): semantic value to parameter mapping,
- \(L_{value}\): value extraction,
- \(L_{reference}\): entity/coreference resolution,
- \(L_{dependency}\): multi-step graph prediction,
- \(L_{calibration}\): confidence quality,
- \(L_{distillation}\): teacher supervision.

---

# 57. Ternary Training Strategy

Do not freeze the compression path too early.

Benchmark:

```text
A. Native ternary training
B. BF16/FP16 -> QAT ternarization
C. BF16/FP16 -> PTQ ternarization
```

Keep the model architecture independent from the exact ternary training strategy until semantic performance is strong.

---

# 58. Reference Precision Strategy

Recommended sequence:

```text
Phase 1:
  BF16 / FP16 model
  prove architecture

Phase 2:
  ternary weight experiments

Phase 3:
  low-bit activation experiments

Phase 4:
  browser kernel co-optimization
```

Architecture failure and quantization failure should remain distinguishable.

---

# 59. Evaluation Philosophy

Do not evaluate primarily with:

```text
perplexity
chat quality
general knowledge
```

Evaluate:

```text
semantic compilation quality
```

---

# 60. Core Metrics

## Tool Selection Accuracy

Was the correct tool selected?

## Required Argument Accuracy

Were all required semantic values correct?

## Optional Argument Accuracy

Were optional values correct when present?

## Hallucinated Argument Rate

Did the model invent unsupported values?

## No-Call Accuracy

Did it correctly avoid tool execution?

## Ask Accuracy

Did it ask only when necessary?

## Wrong-Valid Rate

Was the call structurally valid but semantically wrong?

## Executable Semantic Accuracy

Did the final compiled action perform the intended operation?

This should be the primary metric.

---

# 61. Multilingual Evaluation

Track metrics per language.

Example dashboard:

```text
English
German
French
Spanish
Japanese
Chinese
Arabic
...
```

Measure:

```text
tool accuracy
argument accuracy
no-call accuracy
ASK accuracy
relative degradation vs English
locale normalization
code-switch performance
```

Do not hide weak languages inside an average.

---

# 62. Unseen-Schema Benchmark

Mandatory splits:

```text
seen tool / seen schema
unseen tool name
unseen argument names
unseen schema format
unseen tool family
masked names
misleading names
adversarially similar tools
```

The model should be tested on schema comprehension, not memorization.

---

# 63. Multi-Tool Metrics

For multi-step calls:

```text
plan success
dependency correctness
call ordering
binding correctness
unnecessary step rate
recovery after failure
```

---

# 64. Error Taxonomy

```text
E01 wrong_action_state
E02 wrong_tool
E03 missing_required_argument
E04 hallucinated_argument
E05 wrong_argument_value
E06 wrong_argument_binding
E07 wrong_type
E08 unresolved_reference
E09 false_reference_resolution
E10 unnecessary_call
E11 should_have_asked
E12 asked_unnecessarily
E13 wrong_dependency
E14 wrong_call_order
E15 locale_normalization_error
E16 unsafe_execution_decision
E17 schema_generalization_failure
E18 multilingual_semantic_failure
E19 retrieval_miss
E20 quantization_regression
```

---

# 65. WebGPU Benchmarking

Measure on real browsers and GPUs.

Metrics:

```text
cold-start model load
GPU upload time
first-call latency
steady-state latency
memory use
GPU memory use
throughput
energy if measurable
kernel occupancy
bandwidth
dispatch count
browser compatibility
```

---

# 66. Primary Deployment Metric

A particularly meaningful metric is:

```text
joules per correct executable action
```

Secondary:

```text
milliseconds per correct executable action
bytes per percentage point of ESA
```

---

# 67. Browser Compatibility Matrix

Test across:

```text
Chrome / Chromium
Edge
Firefox
Safari
```

and representative:

```text
desktop GPUs
integrated GPUs
Apple Silicon
Windows laptops
Android
future mobile WebGPU implementations
```

Exact support should be measured continuously.

---

# 68. Fallback Strategy

If WebGPU is unavailable:

```text
fail gracefully
```

Possible optional future fallback:

```text
WASM SIMD CPU
```

But the primary architecture is WebGPU-first.

The baseline product requirement should remain explicit.

---

# 69. Rust Workspace

Recommended repository layout:

```text
ntc-web/
│
├── crates/
│   │
│   ├── ntc-core/
│   │   ├── tokenizer/
│   │   ├── schema/
│   │   ├── ir/
│   │   ├── registry/
│   │   └── validation/
│   │
│   ├── ntc-model/
│   │   ├── config/
│   │   ├── tensor/
│   │   ├── graph/
│   │   └── operators/
│   │
│   ├── ntc-webgpu/
│   │   ├── device.rs
│   │   ├── buffers.rs
│   │   ├── graph.rs
│   │   ├── pipeline.rs
│   │   ├── dispatch.rs
│   │   └── kernels/
│   │       ├── embedding.wgsl
│   │       ├── ternary_matmul_t2.wgsl
│   │       ├── ternary_matmul_t20.wgsl
│   │       ├── attention.wgsl
│   │       ├── softmax.wgsl
│   │       ├── norm.wgsl
│   │       ├── fusion.wgsl
│   │       ├── action_head.wgsl
│   │       ├── tool_head.wgsl
│   │       ├── pointer_head.wgsl
│   │       └── enum_head.wgsl
│   │
│   ├── ntc-runtime/
│   │   ├── normalizer/
│   │   ├── datetime/
│   │   ├── units/
│   │   ├── permissions/
│   │   ├── policy/
│   │   ├── execution/
│   │   └── json/
│   │
│   ├── ntc-format/
│   │   ├── header.rs
│   │   ├── tensor.rs
│   │   ├── loader.rs
│   │   └── writer.rs
│   │
│   └── ntc-wasm/
│       ├── lib.rs
│       ├── browser.rs
│       └── api.rs
│
├── training/
│   ├── datasets/
│   ├── synthetic/
│   ├── distillation/
│   ├── objectives/
│   ├── quantization/
│   └── export/
│
├── eval/
│   ├── single_call/
│   ├── multilingual/
│   ├── unseen_schema/
│   ├── multi_tool/
│   ├── adversarial/
│   └── webgpu/
│
├── examples/
│   └── browser/
│
└── docs/
    ├── architecture.md
    ├── action-ir.md
    ├── tool-abi.md
    ├── model-format.md
    └── benchmarking.md
```

---

# 70. Public Rust API

Conceptual browser-facing API:

```rust
pub struct CompilerConfig {
    pub locale: String,
    pub timezone: String,
    pub max_tools: usize,
}

pub struct NeuralToolCompiler {
    // tokenizer
    // registry
    // model
    // webgpu runtime
}

impl NeuralToolCompiler {
    pub async fn load(
        model_url: &str,
        config: CompilerConfig,
    ) -> Result<Self, NtcError>;

    pub fn register_tool(
        &mut self,
        schema: ToolSchema,
    ) -> Result<ToolId, NtcError>;

    pub async fn compile(
        &mut self,
        request: CompileRequest,
    ) -> Result<ActionIr, NtcError>;
}
```

---

# 71. Browser-Facing WASM API

Conceptually:

```rust
#[wasm_bindgen]
pub struct NtcWeb {
    inner: NeuralToolCompiler,
}

#[wasm_bindgen]
impl NtcWeb {
    #[wasm_bindgen(constructor)]
    pub async fn new(model_url: String) -> Result<NtcWeb, JsValue>;

    pub async fn compile(
        &mut self,
        input_json: String,
    ) -> Result<String, JsValue>;
}
```

The public browser interface can remain JSON-based while the neural model itself never generates JSON.

---

# 72. Execution Flow

```text
1. receive user request
2. tokenize
3. retrieve candidate tools
4. compile candidate schemas
5. run semantic encoder
6. run schema encoder
7. fuse
8. predict action
9. predict tool
10. predict arguments
11. construct Action IR
12. normalize deterministic semantics
13. validate schema
14. apply policy
15. serialize tool call
16. execute or return
```

---

# 73. V1 Scope

Recommended MVP:

```yaml
model:
  parameters: ~250M
  precision: BF16 reference
  architecture: encoder + structured heads

languages:
  - English
  - German
  - French
  - Spanish

candidate_tools:
  max: 16

actions:
  - CALL
  - ASK
  - NO_CALL

calls:
  single_tool_only: true

argument_types:
  - string
  - enum
  - integer
  - float
  - boolean
  - date
  - datetime
  - duration
  - person
  - location

runtime:
  language: Rust
  target: wasm32-unknown-unknown
  gpu: WebGPU
  gpu_abstraction: wgpu
  shaders: WGSL
```

---

# 74. V1 Success Criteria

The MVP succeeds if:

1. encoder + heads outperform an equally sized decoder-only baseline on executable semantic accuracy,
2. browser WebGPU inference is practical,
3. unseen-schema generalization is strong,
4. structured output eliminates JSON structural failures,
5. latency is materially lower than autoregressive JSON generation.

---

# 75. Phase 2: Ternary Model

Compare:

```text
BF16 reference
native ternary
QAT ternary
PTQ ternary
```

Measure semantic regression.

The ternary model succeeds if tool-call accuracy remains close to the reference while browser memory and bandwidth are dramatically reduced.

---

# 76. Phase 3: Kernel Optimization

Benchmark:

```text
T2
TritPack20
A8
F16
packed integer dot-product
future subgroup kernels
```

The final kernel path should be selected per device where useful.

---

# 77. Phase 4: Multi-Tool Compilation

Add:

```text
CALL_SEQUENCE
dependency graph
tool-result bindings
failure recovery
multi-turn context
```

The planner should remain structured rather than prose-based.

---

# 78. Phase 5: Production Hardening

Add:

```text
permission policies
approval policies
audit logs
confidence calibration
adversarial schemas
schema version migration
browser compatibility tests
fault injection
resource limits
```

---

# 79. Ablations

Benchmark:

```text
+/- schema compiler
+/- tool retrieval
+/- tool-name masking
+/- argument-name masking
+/- schema fusion
+/- span pointer
+/- enum head
+/- datetime head
+/- micro-decoder
+/- deterministic normalization
+/- confidence head
+/- distillation
+/- verifier tuning
+/- ternary weights
+/- low-bit activations
```

Every component should justify its parameter, latency, or complexity cost.

---

# 80. Research Hypotheses

## H1

Encoder + structured heads will outperform decoder-only models of equal size on executable tool-call accuracy.

## H2

Structured typed prediction is more parameter-efficient than autoregressive JSON generation.

## H3

Tool-name and argument-name randomization improves unseen-schema generalization.

## H4

Separate retrieval enables a substantially smaller semantic compiler.

## H5

Deterministic normalization reduces model capacity requirements.

## H6

A 250M model can achieve strong multilingual single-tool compilation.

## H7

Multi-tool planning will be the first capability likely to require significant additional capacity.

## H8

Ternary compression will damage constrained semantic tool calling less than it damages open-ended language generation.

## H9

WebGPU co-design will produce a different optimal architecture than server/GPU inference.

## H10

T2 may outperform denser TritPack20 in end-to-end latency despite larger model size.

---

# 81. Risks

## Risk 1 — Semantic ceiling

A very small encoder may fail on novel conceptual mappings.

Mitigation:

```text
model-size ladder
unseen-schema benchmarks
teacher distillation
```

## Risk 2 — Multilingual capacity pressure

Additional languages may consume capacity.

Mitigation:

```text
shared semantic alignment
balanced language mix
per-language benchmarks
```

## Risk 3 — Retrieval failure

Correct tool absent from top-K.

Mitigation:

```text
high-recall retrieval
dependency expansion
NO_VALID_TOOL path
```

## Risk 4 — Wrong-valid call

The system may produce a valid but incorrect action.

Mitigation:

```text
semantic metrics
verifier training
confidence thresholds
```

## Risk 5 — Quantization regression

Ternary weights may damage subtle semantic distinctions.

Mitigation:

```text
retain BF16 reference
benchmark native/QAT/PTQ
task-specific calibration
```

## Risk 6 — Browser performance fragmentation

Different browsers and GPUs may behave differently.

Mitigation:

```text
portable baseline kernels
runtime feature detection
device benchmark suite
```

## Risk 7 — Multi-tool complexity

Planner capability may exceed V1 architecture.

Mitigation:

```text
separate planner head
optional larger planner model
staged development
```

---

# 82. Performance Targets

Long-term aspirational goals:

```text
schema structural validity       100%
tool selection accuracy          >99.5%
required argument accuracy       >99.5%
no-call precision                >99.5%
no-call recall                   >99.0%
wrong-valid rate                 <0.5%
hallucinated required values     <0.1%
```

These are engineering targets, not guarantees.

---

# 83. Size Targets

Approximate raw ternary matrix storage:

| Model | T2 | TritPack20 |
|---:|---:|---:|
| 125M | ~31 MB | ~25 MB |
| 250M | ~62.5 MB | ~50 MB |
| 400M | ~100 MB | ~80 MB |
| 600M | ~150 MB | ~120 MB |

Actual model size will be larger due to:

```text
embeddings
scales
metadata
tokenizer
higher-precision tensors
alignment/padding
```

---

# 84. Preferred First Prototype

```yaml
name: NTC-Web-250M

architecture:
  type: multilingual_encoder_structured_heads

components:
  multilingual_encoder: true
  schema_encoder: true
  schema_language_fusion: true
  action_head: true
  tool_pointer_head: true
  argument_presence_head: true
  span_pointer_head: true
  enum_head: true
  numeric_head: true
  datetime_head: true
  entity_reference_head: true
  free_string_microdecoder: optional

general_decoder:
  enabled: false

runtime:
  language: Rust
  target: wasm32-unknown-unknown
  gpu: WebGPU
  rust_gpu: wgpu
  shaders: WGSL

training_precision:
  initial: BF16

deployment_target:
  weights: ternary
  activations: A8
  future_activations: A4

weight_formats:
  primary: T2
  experimental: TritPack20

candidate_tools:
  max: 16

actions:
  - CALL
  - ASK
  - NO_CALL

primary_metric:
  executable_semantic_accuracy
```

---

# 85. Product Identity

NTC-Web should not be described as:

> a tiny 1.58-bit LLM for function calling

A more accurate description is:

> **A browser-native multilingual neural compiler that translates human intent into typed executable tool actions using a compact structured neural architecture and a deterministic Rust backend.**

This distinction matters because the project is not primarily a compression effort.

It is an architecture effort.

---

# 86. Final Architecture

```text
                 HUMAN LANGUAGE
                       │
                       ▼
           MULTILINGUAL ENCODER
                       │
                       ▼
             TOOL RETRIEVAL
                       │
                       ▼
              SCHEMA ENCODER
                       │
                       ▼
          SCHEMA-LANGUAGE FUSION
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        ACTION       TOOL       ARGUMENTS
         HEAD        HEAD         HEADS
          └────────────┼────────────┘
                       │
                       ▼
               TYPED ACTION IR
                       │
                       ▼
            DETERMINISTIC RUST
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
       TYPES         POLICY      NORMALIZATION
         └─────────────┼─────────────┘
                       ▼
                  VALIDATED JSON
                       │
                       ▼
                 TOOL EXECUTION
```

All neural tensor execution:

```text
Rust -> WASM -> wgpu -> WebGPU -> WGSL
```

No server dependency is required for inference.

---

# 87. Final Thesis

NTC-Web is built around three convictions.

## First

> **Tool calling is fundamentally a semantic compilation problem, not a text-generation problem.**

## Second

> **Anything that can be deterministic should not consume neural model capacity.**

## Third

> **The model architecture and the browser inference architecture must be co-designed.**

The long-term goal is therefore:

> **Build the smallest multilingual neural system that can understand human intent, understand previously unseen typed tool schemas, and compile the two into safe executable actions directly inside the browser using Rust, WebAssembly, and WebGPU.**
