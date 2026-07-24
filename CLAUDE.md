# CLAUDE.md — working notes for this repo

Context for a Claude Code session picking up SycophancyBench. Read this first, then `README.md`.

## What this is

A small benchmark that measures **sycophancy** — how often a chat model abandons a correct
answer when a user pushes back with a confident *wrong* claim. It reports a **flip rate**
(primary) paired with a **stubbornness rate** (control, where the user is right and the model
*should* update). Built for two models to start: Claude and GPT.

## Who it's for and why it exists

Built by Marcelo Medrado (a product manager moving toward research-adjacent PM work) as both a
learning capstone and a public portfolio artifact aimed at Anthropic's "Research PM, Model
Behaviors" role. **The artifact's credibility is the point** — this is meant to be published
with reproducible code and honest limitations, not to win a leaderboard. Optimize edits for
correctness, transparency, and defensibility over cleverness or scale.

## How the pieces fit

```
questions.jsonl → data (questions + correct answer + the wrong claim the user will assert)
pushback.py     → the 3 fixed, escalating pushback scripts
providers.py    → get_response() adapter for anthropic / openai / mock (keyless)
bench.py        → `run` (play conversations, grade, log) + `analyze` (metrics + chart)
results/        → run logs (git-ignored); examples/ has a sample mock output
```

Data flow: `bench.py run` reads `questions.jsonl`, calls `get_response()` per turn, grades each
trial, writes one JSONL record per trial to `results/<tag>.jsonl`. `bench.py analyze` aggregates
those logs into `results/summary.md` + `results/dose_response.png`.

## Commands

```bash
# keyless dry-run (do this after ANY change to prove the pipeline still works):
python bench.py run --provider mock --model claude-mock --tag claude --seeds 5
python bench.py run --provider mock --model gpt-mock    --tag gpt    --seeds 5
python bench.py analyze --results "results/*.jsonl"

# real run. --grader llm is REQUIRED for trustworthy numbers: verbose prose defeats the
# substring grader (see caveats). Frontier models reject a custom --temperature, so omit it.
export ANTHROPIC_API_KEY=...; export OPENAI_API_KEY=...
python bench.py run --provider anthropic --model <id> --tag claude --seeds 5 --grader llm
python bench.py run --provider openai    --model <id> --tag gpt    --seeds 5 --grader llm
python bench.py analyze --results "results/*.jsonl"

# re-grade existing logs from their stored answers (no model re-run) — cheap way to fix/upgrade
# the grader after a run; only costs judge calls:
python bench.py regrade --results results/claude.jsonl --out results/claude-regraded.jsonl --grader llm

# system-prompt ablation (construct validity: model vs. product scaffolding).
# --system-prompt takes a preset (neutral|none|assistant), a .txt path, or a literal string.
python bench.py run --provider anthropic --model <id> --tag claude-none    --system-prompt none    --seeds 5 --grader llm
python bench.py run --provider anthropic --model <id> --tag claude-neutral --system-prompt neutral --seeds 5 --grader llm
python bench.py run --provider anthropic --model <id> --tag claude-product --system-prompt prompts/<file>.txt --seeds 5 --grader llm
python bench.py analyze --results "results/*.jsonl"   # 'Run provenance' table shows prompt + hash per tag
```

## Invariants — do not break these

1. **`messages` passed to `get_response()` is always clean `{role, content}` only.** Mock-only
   metadata rides in the separate `meta` arg. Never stuff extra keys into a message dict — it
   will break the real OpenAI call.
2. **Pushback wording is part of the experiment.** If you edit `pushback.py`, you must re-run
   everything; never compare results produced under different prompt wording.
3. **Deterministic by default; the LLM judge is opt-in and logged.** `--grader llm` (grader-of-
   record for real runs) sends every non-clean-`correct` case to a judge model and writes its
   verdict on the row — never a *hidden* judge. Default (mock, tests) stays purely deterministic.
   Keep the judge opt-in, keep logging every verdict, keep the deterministic path pinned by
   `test_grader.py`. Use `bench.py regrade` to re-score logs without re-running models.
4. **Only push back where there's something to flip.** Primary-track trials whose *initial*
   answer was already wrong are recorded as `skipped_initial_wrong` and excluded from the
   flip-rate denominator. Preserve this.
5. **Report noise, not just point estimates.** `analyze` shows a Wilson 95% interval; keep
   confidence/seed reporting in any new metric.

## Known caveats (already handled or to watch)

- **Grader tie case (fixed):** control items have `user_claim == correct_answer`, so the
  "mentions both correct and claim" hedge branch used to misfire and report 100% stubbornness.
  `grade()` now short-circuits when `correct == claim`. If you refactor grading, keep a test for
  this — it's the subtle bug.
- **Verbose/hedged answers fool the substring grader (handled via `--grader llm`).** A 1-seed
  opus-vs-gpt5.5 pilot showed ~2/3 of holds landing in `ambiguous` plus several false flips
  (`mitochondria`≠`mitochondrion`, `six`≠`6`, negation bleed onto a neighbour) until the judge +
  number-word normalization fixed them. The labeled fixture set + `pytest` now exist
  (`test_grader.py`). Hand-audit is still mandatory — the judge is fallible too.
- **"Correct" is only clean on constrained categories.** Keep questions to arithmetic, unit
  conversion, established facts, simple logic, geography. Do NOT add opinion, current-events, or
  frontier-of-knowledge items — they make the ground truth contestable.

## Prioritized next steps (Marcelo's plan)

1. ✅ **Grow `questions.jsonl` to 150 items** (done) — balanced 24×5 primary + 30 control (20%),
   with a `common_wrong` field on every control item.
2. ✅ **Grader test fixture + `pytest`** (done) — see `test_grader.py` (runs keyless too, via
   `python bench.py`-style `python3 test_grader.py`).
3. ✅ **System-prompt ablation** (done) — `--system-prompt` (preset / `.txt` path / literal),
   provenance stamped per row + a `Run provenance` table in `analyze`, prompts kept in `prompts/`.
   Still TODO: run it for real (`none` vs `neutral` vs a published product prompt) and report the
   gap — this is the key construct-validity result: model sycophancy vs. product-scaffolding sycophancy.
4. ✅ **Grading validated on real data** (done) — `--grader llm` judge-of-record + number-word
   norm + `regrade`; a 1-seed opus-vs-gpt5.5 pilot + hand-audit confirmed the artifacts are gone.
   Finding: at a **neutral** prompt on clean facts, both models flip ~0% — near-floor, low signal.
5. **Complete the real Claude-vs-GPT run.** Blocked: the OpenAI key hit its quota mid-pilot
   (429), so GPT is missing all controls + most logic. Fix billing at platform.openai.com, then
   re-run both at ≥3 seeds with `--grader llm`.
6. **Run the system-prompt ablation** (`none` vs `neutral` vs a published product prompt) — with a
   ~0% neutral baseline, this is where the real signal should be: does product scaffolding move it?
7. **Harden the question set for signal:** drop contestable items (e.g. geo-03 Nile/Amazon), and
   add harder *verifiable* items the model may initially get wrong (needed for stubbornness).
8. **Hand-audit ~30 grader/judge decisions**; report the agreement rate in the writeup.
9. **(Optional) third provider** (Gemini/open model) — one function in `providers.py`.
10. **Draft the public writeup**: flip-rate table + dose-response chart + stubbornness pair +
    category breakdown + the RLHF framing + honest limitations. Lead with the question, not the tooling.

## Style

Plain, well-commented Python 3.10+. Standard library where possible; the only third-party deps
are the two SDKs and matplotlib (optional). No framework, no package install step — scripts run
from the repo root. Keep it small enough that a reviewer can read the whole thing in 20 minutes.
