# Canonical Tool ABI v1

Single implementation: `crates/ntc-core/src/schema/mod.rs`
(`compile_schema` + `CanonicalTool::to_neural_text`). Python consumes it via
`ntc schemac` (JSONL in → JSONL out) — there is deliberately no second
implementation of the rendering, eliminating training/serving skew by
construction. Golden corpus: `fixtures/schema-abi/*.{json,canon.txt}`.

## Accepted raw forms (spec §39)

- Flat NTC style: `parameters: { name: {type, description?, format?, required?, enum?, semantic?} }`
- JSON-Schema / OpenAI function style: `parameters: {type: "object", properties: {...}, required: [...]}`
- Extended explicit types: `person`, `location`, `duration`, `date`, `datetime`, `float`.

Semantic annotations are accepted as `semantic`, `semantic_type`, or `x-semantic`.

## Type mapping

| raw | canonical TYPE |
|---|---|
| string | TEXT |
| string + format date / `date` | DATE |
| string + format date-time / `datetime` | DATETIME |
| string + format duration / `duration` | DURATION |
| integer | INTEGER |
| number / `float` | FLOAT |
| boolean | BOOLEAN |
| any + enum values | ENUM |
| `person` | PERSON |
| `location` | LOCATION |
| `array` + scalar `items` | LIST (+ `ITEM <TYPE>` line) |
| `object` with declared scalar `properties` | flattened into `parent.child` args |
| `array` of objects, free-form `object` | OPAQUE (agent-only) |

Semantic annotations never change TYPE (spec §14 renders `TYPE TEXT` +
`SEMANTIC LOCATION.DESTINATION` as separate facts).

## Rendering grammar (byte-stable per ABI version)

```
TOOL <candidate_index>
DESC <normalized tool description>
ARG <k> <name>
INFO <normalized arg description>     # only if non-empty
TYPE <TYPE>
ITEM <TYPE>                           # only for TYPE LIST (element type)
REQUIRED 0|1
SEMANTIC <SEMANTIC_TYPE>              # only if annotated
ENUM <j> <normalized value>           # one line per enum value
```

## Composite value types (ABI v2, spec §19)

Provider schemas routinely contain arrays and objects. They resolve in three
tiers, cheapest first, so neural capacity is spent only where it must be:

1. **`LIST<scalar>`** — `array` whose `items` declare a scalar type. Rendered
   as `TYPE LIST` + `ITEM <TYPE>`. The model marks **one span** covering the
   list region ("42, 55 and 101"); `ntc-runtime`'s deterministic splitter
   handles separators and multilingual conjunctions and parses each element
   by the declared item type. No list-specific head exists — separators are
   exact rules (spec §6.2).
2. **Object with declared scalar properties** — flattened at compile time into
   dotted pseudo-arguments (`options.published`, `options.note`). Prediction
   stays scalar; the backend re-nests them into the declared object when
   serializing the call.
3. **`OPAQUE`** — free-form objects, arrays of objects, nested payloads. The
   tool stays a candidate (the model may still select it), but
   `CanonicalTool::requires_agent()` is true, and a CALL selecting it is
   converted to `DELEGATE` by the confidence policy: no single typed call can
   express the request, so it belongs to an LLM agent
   (see [delegation.md](delegation.md)).

Determinism rules: NFC normalization; whitespace runs collapsed to single
spaces; lowercased; exactly one trailing `.` stripped; descriptions capped at
200 chars on a word boundary; argument order = declaration order (serde_json
`preserve_order`); lines joined with `\n`, no trailing newline; no version
line in the token stream (the ABI version lives in `.ntc` metadata and the
`CanonicalTool.abi_version` field).

Limits (V1): ≤16 args/tool (model config), ≤12 enum values/arg, non-string
enum values rejected.

## Facts carried but not rendered

Two fields travel on the ABI record without appearing in the token stream, so
declaring them changes nothing the model sees and costs no retrain:

- `risk` (`READ`/`WRITE`/`DESTRUCTIVE`) — for the Phase-5 policy layer.
- `default` — the value the provider applies when the argument is omitted.
  When the source head says the value was constructed rather than taken from
  the request, the deterministic backend supplies this instead of hunting for
  a span. `list_assets.parentId` is the motivating case: 218 corpus rows want
  `1`, the schema's own description says "Default: 1 (root folder)", and the
  model does not need to be taught a constant it can be handed.

Rendering either would change the schema token stream and invalidate every
trained checkpoint, which is the point of keeping them out. Golden fixture
`fixtures/schema-abi/012-default-and-csv-list.*` pins that.

## Semantic annotations that dispatch

`SEMANTIC` is rendered, so adding one *is* a retrain. Three are load-bearing at
decode time rather than merely advisory:

| annotation | effect |
|---|---|
| `…DURATION…` | an INTEGER/FLOAT argument decodes as a duration (spec §40/§44) |
| `LIST.CSV` | a TEXT argument whose provider wants a comma-separated list: the span covers the list region and the deterministic splitter rejoins it in the provider's separator. `get_area_brick.brickIds` — "the video-embed and spec-table bricks" — is the case; typing it `array` would make the backend emit a JSON array the provider rejects. |
| `FILTER.*` | the argument may take one of the model's declared value templates (head codec v4). See [head-codec.md](head-codec.md). |

## Tool ABI record

See `contracts/tool-abi/v1/tool-abi.schema.json` (generated from
`CanonicalTool`). `risk` (`READ`/`WRITE`/`DESTRUCTIVE`, default WRITE) is
carried for the Phase-5 policy layer; V1 does not gate on it.
