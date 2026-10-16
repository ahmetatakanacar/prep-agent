# Example run

A real run of the Titanic benchmark dataset, made with:

```powershell
$env:PREP_AGENT_MODEL = "gpt-6-luna"
uv run prep-agent prepare benchmarks/datasets/titanic.csv --goal "Predict which passengers survived" --target survived --non-interactive
```

`--non-interactive` answered every question with the recommended option (see `qa.json`). The files here are `report.md`, `plan.json`, `qa.json`, `llm_trace.json`, `run.json`, `baseline.json`, `issues.json` and `pipeline.py`.

`train.csv`, `test.csv`, `profile.json` and `pipeline.joblib` are left out. The dataset is downloaded with `benchmarks/download.py`, and `pipeline.py` makes the same train and test files from it.
