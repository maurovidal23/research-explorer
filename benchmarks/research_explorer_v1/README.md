# Research Explorer Fixed Benchmark v1

Status: corpus frozen; question bank generation in progress. This draft must not be
reported as a completed benchmark until all 2,500 items pass deterministic and model
critic validation and `bank_sha256` is non-empty.

This benchmark is the stable comparison set for changes to retrieval, research
memory, and answer quality. It contains 25 pinned arXiv papers and targets 100
validated questions per paper (2,500 total).

The primary score is exact-set accuracy. A response is correct only when the set of
selected option IDs exactly equals the answer-key set. Jaccard partial credit is
reported diagnostically and must not replace the primary score.

Each paper has the same blueprint:

| Dimension | Required distribution |
|---|---:|
| Concepts and mechanisms | 15 |
| Methods and experimental design | 25 |
| Results and interpretation | 20 |
| Assumptions and limitations | 15 |
| Cross-paper distinctions | 15 |
| Transfer and counterfactuals | 10 |
| Medium / hard / adversarial | 20 / 60 / 20 |
| One / two / three correct options | 60 / 30 / 10 |

Every question has five options. Questions may have one, two, or three correct
answers, and the prompt never reveals how many. Every key requires a page-level
evidence locator and a short quote. Distractors should use a nearby paper's method,
reverse a comparison, alter an experimental condition, overgeneralize a result, or
confuse an ablation with the complete system. `all of the above` and `none of the
above` are prohibited.

`manifest.json` pins the source versions. `bank.private.jsonl` is the canonical bank
with keys, rationales, and evidence. A public test payload must be derived with
`BenchmarkQuestion.public_payload()` and must never expose the private fields.

Source PDFs and extracted text are build inputs and are not committed. Acquisition
must verify the hashes recorded in the manifest. The generator is resumable and the
final bank is accepted only when `validate_suite` reports no issues and its canonical
hash matches the manifest.

Build commands, run from the repository root:

```bash
uv run python scripts/build_fixed_benchmark.py acquire
uv run python scripts/build_fixed_benchmark.py generate --batch-size 20
uv run python scripts/build_fixed_benchmark.py critique --batch-size 10
uv run python scripts/build_fixed_benchmark.py prune
```

Repeat `generate`, `critique`, and `prune` until no critic rejection remains, then:

```bash
uv run python scripts/build_fixed_benchmark.py assemble
uv run python scripts/build_fixed_benchmark.py validate
```

All stages can be limited to one paper with `--paper <paper-id>`. Generated batches
and critic verdicts are checkpointed under `data/fixed_benchmark/`, so interrupted
builds resume without repeating accepted work.

Run one isolated research job per paper with bounded concurrency:

```bash
uv run python scripts/run_fixed_research_batch.py \
  --config config/profiles/opencode_exam.toml \
  --max-concurrent 1 \
  --max-fetches 4 \
  --max-reference-entries 5 \
  --reference-mapping-tokens 4000 \
  --arxiv-period 5
```

Each completed paper stores a private frozen survivor bundle, research report, and
run metadata under `data/fixed_benchmark/runs/papers/<paper-id>/`. Re-running the
command skips completed papers whose source and configuration fingerprints still
match. Use `--retry-failed` to retry only failed or stale work.

The corpus deliberately forms one connected technical neighborhood rather
than sampling unrelated fields. This lets Research Explorer use citations and related
work while still testing distinctions among retrieval, reasoning, memory, agents,
long-context modeling, efficient architectures, and software-engineering agents.
