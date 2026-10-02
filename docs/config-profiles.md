# Configuration Profiles

The repository ships several TOML profiles under `config/profiles/`. They tune the
same typed `Config` model loaded by `research_explorer.config.load_config`; profile
files only need to override the keys that differ from `config/default.toml`.

## Completion budgets

The `[llm]` section separates two completion limits:

- `max_tokens` is the budget for the explorer/integration completion (dossier and
  narrative extraction over acquired paper content).
- `evaluation_max_tokens` is the budget for the structured self-assessment, peer-vote,
  and virgin-judge calls. When it is `0` the shared `max_tokens` is used.

`structured_output_attempts` bounds how many times a schema-constrained evaluation
call is retried when the model returns malformed JSON.

## Raised budgets

Earlier profiles reused a single small `max_tokens` value for both paths. That was
too tight for structured dossier/synthesis extraction, which produced truncated or
unparseable JSON. The profiles now reserve a larger completion budget, and the
evaluation path gets an explicit, separately tunable limit:

| Profile | `max_tokens` | `evaluation_max_tokens` | `structured_output_attempts` |
| --- | --- | --- | --- |
| `config/default.toml` | 2000 | 4000 | 2 |
| `config/profiles/benchmark_test.toml` | 4000 | 4000 | 2 |
| `config/profiles/quick.toml` | 16000 (was 2500) | 16000 | 3 |
| `config/profiles/nan_free.toml` | 16000 (was 2000) | 16000 | 3 |
| `config/profiles/nan_full.toml` | 32000 (was 2000) | 32000 | 3 |
| `config/profiles/experiment.toml` | 32000 (was 6000) | 32000 | 3 |

The bounded `benchmark_test.toml` profile keeps small budgets because it runs
deterministic fake examiner/answer clients and never performs provider or LLM calls.

A longer completion limit is not a substitute for structured memory: the survivor
flow persists typed `PaperDossier`/`LedgerClaim` memory (see
`docs/specs/research-survivor-exam.md`) and only renders prose as a derived view.
