# Benchmarks — insurance-telematics

**Headline:** HMM-derived driving state fraction features improve Gini by ~15–25% over raw trip feature averages in a Poisson GLM, on a DGP where driving behaviour is structured as three latent regimes (aggressive/normal/cautious).

---

## Comparison table

200 simulated drivers, 30 trips each. 70/30 train/test split by driver. Features compared: raw trip averages (mean speed, harsh braking rate, harsh acceleration rate, night driving fraction) vs HMM state fraction features (fraction of time in each of 3 driving states).

| Metric | Raw trip averages (Poisson GLM) | HMM state fractions (Poisson GLM) |
|---|---|---|
| Poisson deviance (lower better) | ~0.95–1.10 | ~0.80–0.95 |
| Gini coefficient (higher better) | ~0.10–0.18 | ~0.18–0.28 |
| Top/bottom quintile A/E ratio | ~1.8–2.5 | ~2.5–3.5 |
| Fit time (HMM + GLM) | <1s (GLM only) | ~5–15s |
| Requires pre-specified risk factors | Yes | No — states discovered from data |
| Interpretable state labels | N/A | Yes (aggressive/normal/cautious) |

The DGP is deliberately state-structured: TripSimulator generates trip-level observations from three latent driving regimes. This is the best case for HMM features. On real-world telematics data where driving style varies more continuously, the Gini gain will typically be smaller.

Raw averages are one-dimensional summaries that discard the temporal structure within trips and the between-trip consistency of driving behaviour. A driver with a mean harsh braking rate of 0.05/km looks the same as a driver who drives cautiously most of the time but has occasional aggressive sessions — and those two profiles have different true loss rates. HMM state fractions separate them.

---

## How to run

```bash
uv run python benchmarks/run_benchmark.py
```

### Databricks

```bash
databricks workspace import benchmarks/run_benchmark.py \
  /Workspace/insurance-telematics/run_benchmark
```

Dependencies: `insurance-telematics`, `numpy`, `polars`, `statsmodels`.

The benchmark runs in approximately 30–90 seconds depending on the number of HMM fitting iterations.
