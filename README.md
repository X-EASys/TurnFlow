# TurnFlow

Multi-turn evaluation with an LLM user simulator and source-bound judges.

[Project page](https://x-easys.github.io/TurnFlow/) · [Dataset](https://huggingface.co/datasets/Buka7yo/TurnFlow) · [Task and scoring guide](docs/benchmark.md)

TurnFlow runs a candidate assistant through a sequence of user goals. The
simulator generates each user message from the scenario's current state. After
the final user message, the assistant gives its final answer; a task-specific
LLM judge extracts facts and a fixed Python rule computes the verdict. The
assistant receives conversation messages, without the simulator's private scenario.

Version **0.4.0** uses the contiguous task IDs **T1–T5**, with one 490-scenario snapshot and one default evaluation
protocol. The dataset is bundled with the Python package, so installation and
offline verification require neither Hugging Face credentials nor a GPU.

| Task ID | Task | Scenarios |
|---|---|---:|
| T1 | Stateful Multi-Constraint Following | 100 |
| T2 | Scope Creep | 79 |
| T3 | Constraint Collision | 100 |
| T4 | Preference Override | 97 |
| T5 | Goal Shift | 114 |

## Install

Use Linux or macOS with Python 3.10 or newer; the pinned environment uses Python 3.12.

```bash
git clone https://github.com/X-EASys/TurnFlow.git
cd TurnFlow
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt .
turnflow verify
```

`verify` checks every source file, simulator binding and judge compilation without
calling a model. The lock file pins the dependency environment. For development,
replace `.` with `-e .` in the install command.

## Configure an endpoint

```bash
cp configs/local-router.example.json configs/local.json
```

The local example uses `http://127.0.0.1:8234/v1` without an API key. You must have
an OpenAI-compatible router running there; the router is supplied separately.

For a hosted endpoint, copy `configs/provider.example.json` instead, change
`api_base`, and set the environment variable named by `api_key_env`:

```bash
read -rsp 'API key: ' TURNFLOW_API_KEY
echo
export TURNFLOW_API_KEY
```

Each role can override `api_base`, `api_key_env`, `timeout`, and `retries`. Store
keys in environment variables. Configuration files accept the variable's name,
never a literal key.

The default protocol uses **Gemini 3.7 Flash** for simulation and **DeepSeek V4
Flash** with high reasoning for one judge call per completed dialogue. Candidate
assistants are configurable with `agent.model` or `--agent-model`; set their
`reasoning_effort` and `omit_temperature` in the config as required by the provider.
This release keeps the simulator and judge model names fixed to make the protocol
explicit. Provider availability and costs depend on your endpoint.

Returned model IDs are checked against the requested family. Dated variants of
the same family are accepted; generic names such as `deepseek-flash` do not prove
the V4 identity. `allowed_response_models` only permits additional suffixes within
the same registered family. A routing name cannot establish checkpoint equivalence.

## Run five conversations

```bash
turnflow run --config configs/local.json --output runs/smoke --smoke 1
turnflow report --output runs/smoke
```

**`run` makes paid model calls.** `--smoke 1` selects one source per task, in
source-key order. It checks installation and transport; its pass rate is not a
performance estimate. `FAIL` is a valid benchmark outcome.

List sources, run a task, or run the complete pool:

```bash
turnflow list --tasks T4
turnflow run --config configs/local.json --output runs/t4 --tasks T4
turnflow run --config configs/local.json --output runs/full --concurrency 2
```

Use `--source-keys T4/v3/T4_400` for an explicit subset. Add `--prepare-only`
to write the selection and run manifest without API calls.

The same command with `--resume` continues unattempted positions. Completed,
failed, and interrupted positions remain in the run's denominator and are not
silently resampled. Code, data, routes, and selection must match the saved run.
Use a fresh output directory when changing any of them or intentionally rerunning
failed cases. A run directory has an exclusive lock.

HTTP 408/409/429/5xx, connection failures, empty completions and model identity
mismatches can receive at most three additional attempts for the identical request.
Every physical attempt is recorded. Authentication, exhausted quota, or an
unavailable requested model stops further work; a model mismatch that survives
retries also stops the batch.
The upstream router may implement its own retries. Accepted semantic outputs,
including `FAIL`, `U`, and `NA`, are not retried or voted on.

## Read the outputs

Each run writes its manifest, per-source status, complete/partial conversation,
judge output, and API attempts into the directory you supplied. These files may
contain scenario and conversation text; keep them private or review them before
sharing. No output directory is uploaded automatically.

`summary.json` and `positions.csv` report:

- `pass_rate`: PASS / (PASS + FAIL).
- `scoreable_coverage`: (PASS + FAIL) / planned positions.
- `planned_pass_rate`: PASS / planned positions.
- Separate PASS, FAIL, U, NA, INVALID, and unattempted counts.

An incomplete conversation is INVALID and receives no normal benchmark judgment.
`U` means the contract cannot resolve the case; `NA` means the test is not
applicable. [The scoring guide](docs/benchmark.md) explains the evidence windows
and task-specific rules.

## Dataset and versions

[Buka7yo/TurnFlow](https://huggingface.co/datasets/Buka7yo/TurnFlow) contains matching
`scenarios` and `judges` configurations, each with 490 rows in a `test` split.
The packaged source JSON and HF `scenario_json` strings have identical bytes;
`source_sha256` identifies each source. Judge JSON is provided with the matching
runtime contract and must be interpreted by this version's compiler and parser.

```python
from datasets import load_dataset

scenarios = load_dataset("Buka7yo/TurnFlow", "scenarios", revision="v0.4.0", split="test")
judges = load_dataset("Buka7yo/TurnFlow", "judges", revision="v0.4.0", split="test")
```

Install `datasets` separately for this optional example. Pin the GitHub tag and
the HF revision from [release.json](release.json) together. This release renumbers
the rc5-based snapshot without changing scenario content or scoring rules. Source
IDs and their derived hashes change with the numbering. Earlier release tags
retain their original identifiers; finish existing runs with their original
code/data revision and use a new output directory for this version.

## Limits

Simulation and fact extraction use stochastic models. Boundary cases, source
ambiguities, provider errors and model changes can affect results. This release
provides a fixed evaluation protocol; it does not establish universal judge
correctness or invariant model rankings. Report model/endpoint settings, version,
coverage, technical failures, and all five verdict types with performance results.

## Development and project page

```bash
python -m unittest discover -s tests -v
```

The offline tests use constructed inputs and local HTTP mocks. They require no
model credentials. The static project page is `index.html` with `styles.css`;
preview it with `python3 -m http.server 8000` at http://localhost:8000/. GitHub Pages deploys only those site
assets. The page includes the authors, abstract, framework overview, and current
nine-model results. The paper link and citation remain pending until the arXiv
entry is public.

Code is licensed under [MIT](LICENSE). The bundled source scenarios,
source-specific judge definitions and simulator binding data are licensed under
[CC BY 4.0](DATA_LICENSE).
