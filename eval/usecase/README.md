# Studio use-case acceptance tests

Hand-written scenarios in the shape a real Pimcore Studio user would type,
run through the shipping runtime. Deliberately **not** drawn from the training
corpus: `eval/studio_report.py` already measures held-out performance on the
corpus's own distribution, which tells you the model fits its data. These tell
you whether it survives contact with requests nobody templated.

```sh
cargo build --release -p ntc-cli
python3 eval/usecase/run.py --model models/ntc-studio-v1/model.ntc              # WebGPU (target)
python3 eval/usecase/run.py --model models/ntc-studio-v1/model.ntc --backend cpu # parity oracle
```

Exit code is 0 only when every scenario passes, so this works as a CI gate
once the model is good enough to hold one.

Each scenario asserts only what should be stable — action always, tool when
exactly one is defensible, reason when unambiguous, and specific argument
values where the point of the test is *which* value gets bound. Every entry
carries a `why` explaining what it probes, so a failure says something.

## Result on ntc-studio-v1 (5 training epochs)

**15/26**, byte-identical on WebGPU and the CPU oracle — so every failure is
the model, not the backend.

| group | result | reading |
|---|---|---|
| delegate | **7/8** | The escalation boundary works, including the chained search-filter-mutate reference case, in EN/DE/FR. |
| nocall | 3/4 | Chitchat, unsupported capability and out-of-scope are recognized. |
| call | 4/6 | Zero-argument calls are reliable; the failures are all *within-family* tool confusion (`get_asset` vs `list_assets` vs `search_assets`). |
| policy | 1/2 | The resolution hop (POLICY.md §4.4) is learned; utterance-id-over-linked-item (§4.2) is not. |
| ask | 0/2 | Predicts CALL where it should ask — it does not yet notice that a required argument has no source. |
| adversarial | 0/2 | Namespace traps and mention-only phrasing both fool it. |
| robust | 0/2 | A typo or a politeness prefix flips the decision. |

The shape is consistent with the dev-set numbers: action accuracy 0.872 but
tool selection 0.694. Delegation is a whole-utterance judgement and is solid;
discriminating three sibling tools, and noticing an unfillable argument, are
the parts that need more training. Tool selection was still climbing when the
5-epoch run stopped (0.32 → 0.52 → 0.72 → 0.79 → 0.82 on the training-side
metric), so the first thing to try is simply more epochs — not a new idea.
