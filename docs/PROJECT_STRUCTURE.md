# Project Structure

## Root

- `main.py`: entry point for the full pipeline.
- `requirements.txt`: dependency versions.
- `README.md`: project background and goals.

## Code

- `scripts/pipeline/`: runnable step scripts (`step01` to `step06`).
- `src/original/`: original model/predictor implementations.
- `src/pipeline/`: legacy/alternate pipeline scripts kept for reference.

## Data

- `data/raw/`: raw inputs (for example BLAST text).
- `data/interim/`: intermediate outputs from generation/scoring steps.
- `data/processed/`: final candidates, structures, phylogeny, iTOL outputs.

## Models

- `models/AMPGenix/`
- `models/AMPSorter/`
- `models/BioToxiPept/`
- `models/ProteoGPT/`

## Assets

- `assets/figures/`: images and PDF figures.

## Archive

- `archive/`: non-critical historical files moved out of the project root.
