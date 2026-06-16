# Submission Repository Audit - Nature Health v137

Audit date: 2026-06-15

Repository: `undergroundbreakfast/Scientific_Paper`

## Summary

The public repository is suitable as a bounded reproducibility repository if the manuscript and README clearly state that licensed AHA source files cannot be redistributed. The repository should be cited as a partial reproducibility package, not as a complete raw-data archive.

## Checks performed

- Confirmed public repository exists and default branch is `main`.
- Confirmed no raw AHA hospital-level CSV, Excel, Parquet, SQLite, or database dump files are visible in the current public tree.
- Confirmed code obtains database credentials from environment variables rather than hard-coded password strings.
- Confirmed the repository includes code, generated logs, figures, and aggregated tables from the broader hospital AI research program.
- Added a Nature Health v137 landing description and reproducibility note.
- Added `.gitignore` patterns for raw data, database dumps, environment files, and local geospatial caches.

## Residual limitations

- Full end-to-end reproduction requires licensed AHA files and local database configuration.
- Some older outputs in `results/` and `logs/` relate to a separate outcomes/mortality manuscript. They should not be interpreted as Nature Health causal evidence.
- The Nature Health submission should cite a release tag, not only the moving `main` branch.

## Recommended citation language

"Reproducibility scripts, logs, and aggregated derived outputs not restricted by the AHA license are available at `https://github.com/undergroundbreakfast/Scientific_Paper` (release tag `v137-submission`). Licensed AHA Annual Survey records cannot be redistributed."
