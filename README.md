# Scientific_Paper

Public reproducibility repository for the hospital AI adoption, proximity, and outcomes research program.

## Nature Health v137 submission

This repository supports the Nature Health submission:

**Coverage without convergence: Geographic inequality in proximity to AI-reporting hospitals in the United States, 2022-2024**

The Nature Health manuscript links 2022 and 2024 American Hospital Association (AHA) AI-use measures to U.S. Census block groups and county health data to estimate population proximity to hospitals reporting active AI deployment. The submission focuses on population coverage, travel-time proximity, access-distance inequality, transition groups, and pre-diffusion health-burden overlap.

The AHA source files are licensed data and are not redistributed here. This repository is therefore a partial reproducibility package: it documents the analysis logic and provides code, logs, and releasable derived outputs where those outputs do not redistribute restricted AHA records.

See [docs/nature_health_v137_reproducibility_note.md](docs/nature_health_v137_reproducibility_note.md) for the v137-oriented reproducibility boundary, data dependencies, script orientation, and manuscript-output mapping.

## Important scope note

This repository also contains artifacts from related hospital AI, robotics, and mortality/outcomes analyses. Those older outputs are retained for transparency in the broader research program, but they are not the evidentiary basis for the Nature Health access manuscript. The Nature Health paper does not estimate causal mortality effects of AI deployment; it uses Years of Potential Life Lost (YPLL) only as pre-diffusion health-burden context.

## Repository structure

- `code/`: analysis scripts used in the broader hospital AI research program.
- `results/figures/`: generated PNG charts and diagnostic plots.
- `results/tables/`: generated CSV summaries, sensitivity results, and supporting tables.
- `logs/`: run-time logs and plain-text execution traces.
- `docs/`: run memory, reproducibility notes, and supporting documentation.
- `LICENSE`: repository license.

## Key scripts

- `code/Geospatial_Lorenz_Curve_021426_v26.py`: geospatial proximity, population coverage, Lorenz/Gini, and publication-figure workflow underlying the Nature Health access analysis.
- `code/Replicate_Results_090625_v55.py`: county-level outcomes and robustness workflow used in a related outcomes/access manuscript, not as causal evidence in the Nature Health submission.
- `code/Generate_Moderation_Plots_031526_v1.py`: supporting plot-generation workflow for related moderation analyses.

## Data availability limits

Not included:

- Raw AHA Annual Survey hospital-level files.
- Licensed AHA field-layout files beyond public links cited in the manuscript.
- Intermediate hospital-level files that would redistribute licensed AHA records.
- Local database dumps, credentials, or environment files.

Included or intended for public release:

- Reproducibility scripts.
- Logs and generated outputs that do not disclose restricted AHA records.
- Aggregated derived tables and figures where public release is permitted.

Running the full pipeline from raw inputs requires licensed AHA files, public Census and County Health Rankings inputs, and local path/database configuration.

## Environment

The scripts are Python-based and expect a local PostgreSQL database containing licensed AHA-derived tables. Database credentials should be supplied through environment variables, especially `POSTGRESQL_KEY`; credentials should never be committed.

A lightweight dependency list is provided in [requirements.txt](requirements.txt). Exact package versions may vary by local geospatial stack.

## Release hygiene

For journal submission, cite a frozen release tag rather than a moving branch. The intended submission tag is:

```bash
git tag -a v137-submission -m "Nature Health v137 submission reproducibility package"
git push origin main --tags
```

If the submitted manuscript advances beyond v137, create a new tag matching the submitted version.
