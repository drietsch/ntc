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

Semantic annotations never change TYPE (spec §14 renders `TYPE TEXT` +
`SEMANTIC LOCATION.DESTINATION` as separate facts).

## Rendering grammar (byte-stable per ABI version)

```
TOOL <candidate_index>
DESC <normalized tool description>
ARG <k> <name>
INFO <normalized arg description>     # only if non-empty
TYPE <TYPE>
REQUIRED 0|1
SEMANTIC <SEMANTIC_TYPE>              # only if annotated
ENUM <j> <normalized value>           # one line per enum value
```

Determinism rules: NFC normalization; whitespace runs collapsed to single
spaces; lowercased; exactly one trailing `.` stripped; descriptions capped at
200 chars on a word boundary; argument order = declaration order (serde_json
`preserve_order`); lines joined with `\n`, no trailing newline; no version
line in the token stream (the ABI version lives in `.ntc` metadata and the
`CanonicalTool.abi_version` field).

Limits (V1): ≤16 args/tool (model config), ≤12 enum values/arg, non-string
enum values rejected.

## Tool ABI record

See `contracts/tool-abi/v1/tool-abi.schema.json` (generated from
`CanonicalTool`). `risk` (`READ`/`WRITE`/`DESTRUCTIVE`, default WRITE) is
carried for the Phase-5 policy layer; V1 does not gate on it.
