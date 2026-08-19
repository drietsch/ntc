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

## Head-codec / compatibility notes

The action head is append-only: v2 added `DELEGATE` at index 3 and records the
width in `.ntc` `model.action_classes` (default 3). Models trained before the
change load unchanged and simply never predict it (`contracts/VERSIONS.md`).
