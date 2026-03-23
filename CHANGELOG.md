# Changelog

## [0.1.7] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)



## v0.1.6 (2026-03-23)
- fix: pin hmmlearn>=0.3.3 — versions before 0.3.3 used numpy.core.multiarray which was removed in numpy 2.x; 0.3.3 is the first release shipping numpy 2-compatible wheels

## v0.1.4 (2026-03-22) [unreleased]
- fix: correct license badge from BSD-3 to MIT
- fix: use plain string license field for universal setuptools compatibility

## v0.1.4 (2026-03-21)
- docs: replace pip install with uv add in README
- Add blog post link and community CTA to README
- Add MIT license
- Reduce telematics benchmark fleet size to avoid OOM on Databricks serverless
- Add benchmark: HMM state features vs raw trip averages in Poisson frequency GLM
- docs: regenerate API reference [skip ci]
- fix: QA audit batch 5 — accuracy and documentation fixes (v0.1.4)
- Add PyPI classifiers for financial/insurance audience
- Add Colab quickstart notebook and Open in Colab badge
- fix: correct feature column name and relax Q recovery threshold
- docs: regenerate API reference [skip ci]
- Fix P0-1 and P0-2 xi/gamma normalisation bugs in ContinuousTimeHMM._e_step
- docs: regenerate API reference [skip ci]
- pin statsmodels>=0.14.5 for scipy compat
- fix: move pytest to dependency-groups so uv sync installs it by default
- Add shields.io badge row to README
- docs: add Databricks notebook link to burning-cost-examples
- Add Related Libraries section to README
- fix: define trip_features_df and time_deltas in HMM classification code block
- fix: update polars floor to >=1.0 and fix project URLs
