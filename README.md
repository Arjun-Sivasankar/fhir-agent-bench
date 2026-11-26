# fhir-agent-bench

A read-only MCP server over FHIR records of synthetic patients, a LangGraph agent that answers
clinician-style questions through it ("last 10 HbA1c values", "active medications",
"conditions since 2020"), and a benchmark of tool-call accuracy, answer correctness and refusal
behaviour across open-weight and API models.

Work in progress: see the [milestones](https://github.com/Arjun-Sivasankar/fhir-agent-bench/milestones)
and issues. Results appear here only once a run in this repo has produced them.

No real patient data is used: tests run on a small hand-written synthetic FHIR bundle, and the
benchmark uses Synthea synthetic patients (Apache-2.0).

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[agent,dev]"
pre-commit install
python -m pytest
```

Work runs through issues and pull requests: one branch per issue, conventional commits, CI
(ruff + pytest) must pass before a squash-merge.

## Licence

MIT, see [LICENSE](LICENSE). Author: Arjun Sivasankar ([github.com/Arjun-Sivasankar](https://github.com/Arjun-Sivasankar)).
