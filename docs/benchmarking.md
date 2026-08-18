# Benchmarking (spec §65–§66)

## Measured — mini model (ntc-mini-v1, 1.42M params, F16, 2.8 MB), Chrome on Apple M3 Max, 2026-08-18

| Metric | WebGPU | CPU (wasm reference) |
|---|---:|---:|
| Model fetch (localhost) | 13 ms | 13 ms |
| Instance init (decode + upload) | 33–87 ms | ~21 ms |
| First compile (incl. shader compile) | ~164 ms | — |
| Steady-state compile | **33–44 ms** | ~196 ms |

- WebGPU is ~4.5× faster than the wasm CPU path at this size and produces
  byte-identical outcomes (browser-side decision parity).
- Numbers from the demo page's own timers (`examples/browser`); rerun by
  serving the repo root and clicking Compile.

Native eval throughput: `ntc batch-infer` (release) processes the 331-example
dev+test sets in seconds.

## Still to measure (full-scale model)

- Cold-start load + GPU upload at 250M/F16 (~500 MB artifact).
- Joules per correct executable action (primary deployment metric).
- Browser matrix: Edge/Firefox/Safari × GPU classes (spec §67).
