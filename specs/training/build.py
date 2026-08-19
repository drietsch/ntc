# -*- coding: utf-8 -*-
import json, random, sys, collections
sys.path.insert(0, '/home/claude/gen')
import vocab as V
from engine import build_installation, REGISTRY, POOL
import templates_a, templates_b, templates_c

BUILDERS = templates_a.BUILDERS + templates_b.BUILDERS + templates_c.BUILDERS
LANGS = ['en', 'de', 'fr', 'es']
DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 9

rng = random.Random(20260819)
records, seen = [], set()

for builder in BUILDERS:
    for lang in LANGS:
        for _ in range(DRAWS):
            vert = rng.choice(V.VERTICALS)
            inst = build_installation(rng, lang, vert, V)
            try:
                rec = builder(rng, lang, inst, V)
            except Exception as e:
                print('BUILDER FAIL', builder.__name__, lang, repr(e)); raise
            if rec is None:
                continue
            key = (rec['utterance'], json.dumps(rec['context'], sort_keys=True),
                   tuple(sorted(c['name'] for c in rec['candidates'])))
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)

rng.shuffle(records)

# ---------------------------------------------------------------- validation
def spans_of(a):
    out = []
    if 'char_span' in a:
        out.append((a['char_span'], a.get('surface')))
    for e in a.get('elements', []) + a.get('composed_from', []):
        out.append((e['char_span'], e['surface']))
    return out

errors = collections.Counter()
for r in records:
    u = r['utterance']
    g = r['gold']
    refs = {l['ref'] for l in r['context']['linked']}
    names = [c['name'] for c in r['candidates']]
    if len(names) != len(set(names)):
        errors['duplicate_candidate'] += 1
    if g.get('tool') and g['tool'] not in names:
        errors['gold_not_in_slate'] += 1
    for a in g['arguments']:
        for span, surf in spans_of(a):
            if u[span['start']:span['end']] != surf:
                errors['span_mismatch'] += 1
        if a['source'] == 'linked_item':
            rs = a.get('linked_refs') or [a.get('linked_ref')]
            if not set(rs) <= refs:
                errors['bad_linked_ref'] += 1
        if a['source'] == 'utterance' and 'char_span' not in a and 'elements' not in a \
           and 'composed_from' not in a:
            errors['utterance_arg_without_span'] += 1
        tool = g.get('tool')
        if tool and a['parameter'] not in REGISTRY[tool]['parameters']:
            errors['unknown_parameter'] += 1
        if a['semantic_type'] == 'ENUM':
            opts = REGISTRY[tool]['parameters'][a['parameter']]['enum']
            if a['value']['symbol'] not in opts or \
               opts[a['value']['index']] != a['value']['symbol']:
                errors['enum_mismatch'] += 1
    if g['action'] == 'CALL' and g.get('tool'):
        have = {a['parameter'] for a in g['arguments']}
        req = {k for k, v in REGISTRY[g['tool']]['parameters'].items() if v.get('required')}
        if req - have:
            errors['call_missing_required'] += 1
    if g['action'] == 'ASK' and not g['unresolved']:
        errors['ask_without_unresolved'] += 1
    if g['action'] == 'DELEGATE' and not g.get('delegate_reason'):
        errors['delegate_without_reason'] += 1

# --------------------------------------------- grouped split (no leakage)
groups = {}
for r in records:
    groups.setdefault(r['utterance'].strip().lower(), []).append(r)
keys = sorted(groups)
rng.shuffle(keys)
n_dev = max(1, int(0.12 * len(keys)))
dev = set(keys[:n_dev])
for k, rs in groups.items():
    for r in rs:
        r['split'] = 'dev' if k in dev else 'train'

with open('/mnt/user-data/outputs/train.jsonl', 'w', encoding='utf-8') as f:
    for r in records:
        if r['split'] == 'train':
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open('/mnt/user-data/outputs/dev.jsonl', 'w', encoding='utf-8') as f:
    for r in records:
        if r['split'] == 'dev':
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ------------------------------------------------------------------ report
c = collections.Counter
print('records          :', len(records))
print('  train / dev    :', sum(1 for r in records if r['split'] == 'train'),
      '/', sum(1 for r in records if r['split'] == 'dev'))
print('validation errors:', dict(errors) or 'none')
print('action           :', dict(c(r['gold']['action'] for r in records)))
print('lang             :', dict(c(r['lang'] for r in records)))
print('vertical         :', dict(c(r['vertical'] for r in records)))
print('slate size       :', dict(sorted(c(len(r['candidates']) for r in records).items())))
print('distinct slates  :', len({tuple(sorted(x['name'] for x in r['candidates']))
                                 for r in records}))
print('delegate reasons :', dict(c(r['gold'].get('delegate_reason') for r in records
                                   if r['gold']['action'] == 'DELEGATE')))
print('arg sources      :', dict(c(a['source'] for r in records
                                   for a in r['gold']['arguments'])))
print('semantic types   :', dict(c(a['semantic_type'] for r in records
                                   for a in r['gold']['arguments'])))
gold = c(r['gold']['tool'] for r in records if r['gold'].get('tool'))
sugg = c(r['gold'].get('suggested_tool') for r in records if r['gold'].get('suggested_tool'))
print('\ntools as gold / suggested / candidate:')
cand = c(x['name'] for r in records for x in r['candidates'])
for t in sorted(POOL):
    print(f'  {t:33} gold={gold.get(t,0):4} sugg={sugg.get(t,0):4} cand={cand.get(t,0):5}')
never = [t for t in POOL if gold.get(t, 0) == 0 and sugg.get(t, 0) == 0]
print('never gold nor suggested:', never)

# duplicate + shortcut checks
utt = c(r['utterance'].strip().lower() for r in records)
print('\nduplicate utterance groups:', sum(1 for v in utt.values() if v > 1),
      'extra rows:', sum(v - 1 for v in utt.values() if v > 1))
import statistics
for a in ['CALL', 'ASK', 'NO_CALL', 'DELEGATE']:
    L = [len(r['utterance'].split()) for r in records if r['gold']['action'] == a]
    print(f'{a:9} median words {statistics.median(L):5.1f}  n={len(L)}')
