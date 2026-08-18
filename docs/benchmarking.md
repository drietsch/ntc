# Benchmarking (spec §65–§66)

Not yet implemented — lands with milestone A8. Planned measurements:

- cold-start model load, GPU upload time, first-call latency, steady-state
  latency, memory + GPU memory, dispatch count (native: criterion; browser:
  a bench page in examples/browser).
- Primary deployment metric: joules per correct executable action (where
  measurable); secondary: ms per correct action, bytes per ESA point.
- Browser matrix: Chrome/Edge/Firefox/Safari × Apple Silicon/integrated/
  discrete GPUs (spec §67), recorded per run in a results table.
