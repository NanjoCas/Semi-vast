# Copilot Workspace Instructions

This repository is the `Semi-vast/Sophia` research project for semi-supervised fake news detection.

## Primary guidance

- Treat `PROJECT_README.md` as the authoritative architecture and research spec.
- Prefer existing scripts and configuration over new custom pipelines.
- Use `configs/config.yaml` for defaults and hyperparameter values.
- Do not modify raw data under `Data/` or published artifacts under `processed/`, `checkpoints/`, `outputs/`, or `model_cache/` unless the task specifically requires it.

## Key files and directories

- `PROJECT_README.md`: primary agent-facing project spec
- `run_pipeline.py`: data preprocessing and dataset construction
- `training/`: model training and pseudo-label selection scripts
- `configs/config.yaml`: default model, data, training, and RL settings
- `requirements.txt`: Python dependencies for the project
- `Data/`: raw dataset downloads
- `processed/`: generated JSONL datasets for training and inference
- `checkpoints/`: saved model checkpoints
- `outputs/`: evaluation outputs and logs

## Most important commands

- `python run_pipeline.py`
- `python training/generate_pseudolabels.py --config configs/config.yaml`
- `python training/train_extractor.py --config configs/config.yaml`
- `python training/train_rl_selector.py --config configs/config.yaml`
- `python training/train_detector.py --config configs/config.yaml`
- `bash setup_env.sh`
- `bash run_full_pipeline_cuda.sh`

## Workflow expectations

1. Start by understanding the task in `PROJECT_README.md`.
2. Inspect `configs/config.yaml` before editing hyperparameters or model paths.
3. Reuse existing dataset classes and logging patterns instead of inventing new I/O formats.
4. Preserve Chinese comments and naming conventions in the codebase.

## Project conventions

- Dataset files are JSONL with one object per line.
- Label mapping is typically:
  - `SUPPORTS` = 0
  - `REFUTES` = 1
  - `NOT_ENOUGH_INFO` = 2
- The pipeline is research-focused; ensure reproducibility of training and evaluation.

## Notes for the AI assistant

- Link to `PROJECT_README.md` instead of duplicating long design details.
- If a task involves model training, check `configs/config.yaml` and use the existing training scripts.
- If there is ambiguity about dataset structure, consult `run_pipeline.py`.
- Avoid making large or destructive changes without explicit user approval.
