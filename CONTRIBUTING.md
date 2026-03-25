# Contributing to insurance-telematics

This library converts raw telematics trip data into auditable GLM-ready risk scores. The pipeline has to work on real data from real telematics providers, which varies considerably. Contributions that improve coverage of real-world data formats and edge cases are especially valuable.

## Reporting bugs

Open a GitHub Issue. Include:

- The Python and library version (`import insurance_telematics; print(insurance_telematics.__version__)`)
- The data format you are working with (1Hz GPS, accelerometer only, combined, pre-aggregated events)
- A minimal reproducible example — use the synthetic trip data generator if you cannot share real data
- What you expected and what actually happened

Common failure modes worth reporting: HMM convergence issues on short trips, state count sensitivity when the default 3-state model does not fit your provider's data, and aggregation edge cases for drivers with very few trips.

## Requesting features

Open a GitHub Issue with the label `enhancement`. Priority areas: additional telematics provider format parsers, alternative HMM formulations (continuous-time, Bayesian), and integration with common UK telematics data lake schemas.

## Development setup

```bash
git clone https://github.com/burning-cost/insurance-telematics.git
cd insurance-telematics
uv sync --dev
uv run pytest
```

The library uses `uv` for dependency management. Python 3.10+ is required. Tests use synthetic trip data generators and do not require real telematics data.

## Code style

- Type hints on all public functions and methods
- UK English in docstrings and documentation
- Docstrings follow NumPy format and note units explicitly — speeds in km/h, distances in metres, times in seconds unless stated otherwise
- The `TelematicsScorer` class is the primary public interface; keep it stable
- HMM implementation details live in internal modules — contributions that change the public API need a clear rationale

---

For questions or to discuss ideas before opening an issue, start a [Discussion](https://github.com/burning-cost/insurance-telematics/discussions).
