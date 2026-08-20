# The router boundary: CALL / ASK / NO_CALL / DELEGATE

NTC is the **fast local tier** of an agent stack. It answers one question per
utterance — *can this be compiled into a single typed tool call right now?* —
in tens of milliseconds, on-device, with no server round-trip.

```
utterance ─► NTC router (30–90 ms, local)
               ├── CALL      → validated JSON tool call, execute immediately
               ├── ASK       → one required argument missing; ask the user
               ├── NO_CALL   → nothing to execute (mention, chit-chat, unsupported)
               └── DELEGATE  → real work, but beyond a single typed call
                                  ↓
                              full LLM agent (may use the same tools, many turns)
```

## Why DELEGATE is its own verdict

`NO_CALL` means *there is nothing to run*. `ASK` means *one argument away from
a call*. Neither fits a request that is genuinely actionable but structurally
bigger than one call. Collapsing such requests into `NO_CALL` would make the
router look "safe" while silently dropping work; guessing a `CALL` would be
worse — a wrong mutation. `DELEGATE` names the situation and routes it.

The runtime emits `CompileOutcome::Delegate { utterance, candidates, ir }`:
the original utterance plus the candidate tool ids the router saw, so the host
can hand the agent the same context without re-deriving it.

## What belongs in each verdict

| Verdict | Shape of request | Example (Pimcore) |
|---|---|---|
| CALL | one self-contained action, arguments present in the utterance | "show me asset 812" |
| ASK | one required argument missing, intent otherwise unambiguous | "show me the asset" |
| NO_CALL | mention, meta-question, chit-chat, unsupported capability | "what does search_assets do?" |
| DELEGATE | chain where later steps depend on earlier **results**; bulk mutation over a filtered/unknown set; conditional or comparative logic; open-ended authoring/analysis | "Search for all blue cars in Pimcore assets. Then take just the cars that have a building year of 1976 and before and activate the oldtimer field for those cars." |

Mutations deserve particular care: a single-call compiler that is *unsure*
about a write should prefer `DELEGATE` (or `ASK`) over a plausible-looking
`CALL`. The confidence policy already fails closed in that direction.

## Relationship to spec §77 (Phase 4: CALL_SEQUENCE)

Some multi-step requests have a *mechanical* shape — fixed steps with
result-bindings (`step_1.output → step_2.argument`). Spec §77 plans a
structured planner head for exactly those, and it stays local. `DELEGATE`
covers the rest: data-dependent control flow, unbounded fan-out, judgement.
When the planner lands, the boundary moves — some of today's `DELEGATE`
becomes tomorrow's `CALL_SEQUENCE` — without changing this contract.

## The boundary moved once already (v4)

`eval/esa.py` used to count any argument with no span behind it as beyond a
single typed call — 13.8% of Studio's calls, escalated to an LLM by design.
Reading those rows rather than the total, they were a constant the schema
declares, a list whose elements are in the utterance, and 258 PQL filters in
five closed shapes.

So they were never `DELEGATE`'s kind of problem. `DELEGATE` is for requests
whose *shape* is unknown until earlier steps run — data-dependent control flow,
unbounded fan-out, judgement. "list every PDF" has a completely known shape; it
was only unreachable because nothing could fill in a blank. Head codec v4
fills blanks (see [head-codec.md](head-codec.md)), and the Studio dev ceiling
went from 88.8% to 100%.

The lesson generalises: "the compiler cannot express this" and "this needs a
language model" are different claims, and the first was standing in for the
second. Before adding a case to this page, check which one it is.

## Head-codec / compatibility notes

The action head is append-only: v2 added `DELEGATE` at index 3 and records the
width in `.ntc` `model.action_classes` (default 3). Models trained before the
change load unchanged and simply never predict it (`contracts/VERSIONS.md`).
