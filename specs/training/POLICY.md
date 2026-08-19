# Label policy: Pimcore in-browser tool delegator

Every gold label in `train.jsonl` / `dev.jsonl` follows the rules below. Tool
names, descriptions and parameter schemas are extracted from the
PimcoreAgentBundle source (`registry.json`, 49 tools), never hand-written, so
they cannot drift from the bundle.

---

## 1. Record schema

```jsonc
{
  "id": "...",                       // sha1 of template + lang + utterance + context
  "lang": "de",
  "vertical": "pim",                 // pim | dam | mdm | dxp | cdp | ecom
  "template_id": "t_tag_name_resolution",
  "utterance": "tagge das mit Freigegeben",
  "context": {                       // the Studio frame; may be entirely empty
    "linked": [                      // items linked to the chat
      {"ref": "L1", "type": "asset", "id": 4711,
       "key": "winterjacke-hero.jpg", "path": "/Produktfotos/",
       "isFolder": false, "className": null}
    ],
    "selectionCount": 37,            // present when the selection exceeds `linked`
    "resolver": [                    // pre-pass lookup of integer-like tokens
      {"token": "4711", "char_span": {...},
       "candidates": [{"type": "asset", "id": 4711, "key": "..."},
                      {"type": "document", "id": 4711, "key": "..."}]}
    ],
    "locale": "de"
  },
  "candidates": [ {"name": ..., "description": ..., "parameters": {...}} ],
  "gold": { ... },
  "split": "train",
  "tags": ["call", "tagging", "resolution_hop"]
}
```

## 2. Actions

| action | meaning |
|---|---|
| `CALL` | one tool, every required argument derivable now |
| `ASK` | right tool identified, a required argument cannot be filled |
| `DELEGATE` | hand to the large model, with a `delegate_reason` |
| `NO_CALL` | no tool should run |

`DELEGATE` always carries a reason, and usually a `suggested_tool`:

- `PAYLOAD_REQUIRED` — the call needs a nested object/array payload
  (`propose_*`, `update_*`). The delegator never builds those.
- `OVER_LIMIT` — a single intent that exceeds a hard per-call cap
  (`apply_transition`/`apply_global_action`/`get_document` max 5,
  `propose_document_update` max 20, search `pageSize` max 50). Needs a loop.
- `MULTI_STEP` — conjunctive or genuinely chained request.
- `MIXED_ELEMENT_TYPES` — selection spans types a single-type tool cannot take.

`NO_CALL` carries a `no_call_reason`: `CHITCHAT`, `CONCEPTUAL_QUESTION`,
`UNSUPPORTED_CAPABILITY`, `OUT_OF_SCOPE`, `MENTION_ONLY`.

## 3. Argument provenance

Every argument declares a `source`:

| source | carries |
|---|---|
| `utterance` | `char_span` + `surface` (verified: `utterance[start:end] == surface`) |
| `linked_item` | `linked_ref` / `linked_refs` into `context.linked` |
| `resolver` | `resolver_token` |
| `inferred` | neither: constructed (PQL string, `parentId=1`, `includeContent=true`) |

Semantic types: `STRING`, `INTEGER`, `BOOLEAN`, `ENUM`, `ARRAY`.
`ARRAY` carries `item_type` plus per-element spans or per-element linked refs.
`ENUM` stays `{index, symbol}` and is validated against the enum order in that
tool's schema.

## 4. Decisions that were open, and how they are resolved here

1. **Write intents default to the proposal track.** A write is
   `DELEGATE / PAYLOAD_REQUIRED` with `suggested_tool = propose_*`. The direct
   `update_*` track is only suggested when the utterance explicitly authorises a
   small edit ("just set it directly, no review"), marked
   `authorization: EXPLICIT_DIRECT_WRITE`. This is the rule
   `ProposeDocumentUpdateTool` states in its own description.
   Change one flag in the generator if you want the opposite default.
2. **An explicit ID in the utterance beats a linked item.**
   Marked `precedence: UTTERANCE_OVER_LINKED_ITEM`, with `context_used: []`.
3. **"Needed tool absent from slate" is a `CALL` to `discover`**, not a forced
   pick. `execute` and `finalize_task` are excluded from the candidate pool
   entirely: `execute` is redundant whenever the target is in the slate and its
   `arguments` object is not span-derivable; `finalize_task` is turn lifecycle,
   not user intent.
4. **Name-to-ID resolution is a first-class `CALL`.** "Tag this with Freigegeben"
   is `list_tags(filter="Freigegeben")`, not `assign_tag`. Marked
   `resolution_hop_for`.
5. **PQL is constructed, not copied**, and is marked
   `prerequisite: "skill:pql-search"` per the tool descriptions.

## 5. Anti-shortcut construction

- **Length no longer separates DELEGATE.** In the previous file, `len >= 10`
  caught 98% of DELEGATE at 13% false positives. Here the best threshold gives
  0.40 recall at 0.18 FP. Short over-limit bulk ("tag all of these") and long
  single-call utterances are both present.
- **Slates vary**: 2 to 8 candidates, same-group / cross-group / gold-absent,
  ~2075 distinct slates. Gold position is uniform.
- **Spurious binding negatives**: of 1320 rows with a linked item, 792 arguments
  bind it and roughly half the rows deliberately ignore it (e.g. "create a tag
  called X" with an object linked).
- **Identifiers are copied, not recalled**: class names, fields, tags,
  workflows, states, folders, filenames and doc types are drawn per record from
  a synthetic installation, in-language, including umlauts, accents, CamelCase,
  German compounds, and collisions with Pimcore's own nouns (`Seite`, `Link`,
  `Page`, `Kategorie` as customer class names).
- **Surface noise**: prefixes, politeness suffixes and full lowercasing, with
  spans shifted and surfaces adjusted so they stay exact.
- **Split is grouped by utterance**, so duplicates cannot leak into `dev`.

## 6. Adversarial cases deliberately included

| tag | what it tests |
|---|---|
| `namespace_trap` | `parentId` is a tag id in `create_tag`, a document id in `create_document`, an asset folder id in `upload_asset` |
| `family_dependent_symbol` | tag tools use `object`, workflow tools use `data-object` for the same linked item |
| `enum_from_context` | `elementType` from the linked item when no type word appears |
| `ambiguous_namespace` | one integer resolves as both an asset and a document |
| `not_found` | resolver returns nothing (indistinguishable from no permission, by design) |
| `type_conflict` | linked document, asset-only intent |
| `source_conflict` | utterance ID vs linked item |
| `over_max_clamped` | `pageSize` above the server cap |
| `mention_only` | the verb names a tool but the user asks for an explanation |
| `contradicts_search_assets_description` | `SearchAssetsTool` claims `list_assets` does not exist, but it does |

## 7. Known open items

- The proposal-vs-direct-write default (§4.1) is a product decision. Flip it in
  `templates_a.d_update_field_*` and `templates_c.x_*` if it should differ.
- No `targetGroupId` / personalisation rows yet beyond `list_target_groups`.
- `stage_file_set`, `list_editables` and the agent-template tools are developer
  intents; they are represented thinly and may not belong in an end-user slate.
- Chat-session-scoped tools (`propose_*`, `stage_asset`, `stage_file_set`,
  `finalize_task`) are unavailable under PAT / SessionBridge auth, so a
  deployment-specific pool filter may be needed.
