# SycophancyBench

**Measuring belief-capitulation rate across frontier chat models.**

When a language model gives you a correct answer and then abandons it the moment you push
back — even though you're wrong — that's *sycophancy*. It's one of the most consequential
behavioral failure modes in deployed assistants and one of the hardest for a user to notice,
because it fails you precisely when you were already mistaken.

SycophancyBench turns that into a number. It asks a model factual questions it gets right,
has a scripted "user" push back with a confident *wrong* claim at escalating intensity, and
measures how often the model **flips**. It reports the result overall, by category, and as a
**dose-response curve** across pushback intensity — and it pairs the flip rate with a
**stubbornness** control, because a model that never flips might just be rigid rather than
honest.

> Built by Marcelo Medrado as the capstone of a 60-day LLM-research curriculum (Day 59), and
> as a working artifact for research-adjacent PM work on model behavior and evaluation.

---

## Why this is designed the way it is

Sycophancy isn't a random quirk — it's a **predictable consequence of RLHF**. If a reward
model is trained on human preferences and humans reward answers that agree with them, the
optimized policy learns that agreement is rewarded. So the flip rate is partly a *fingerprint
of each lab's preference optimization* and how hard their later training (e.g. Constitutional
AI, explicit anti-sycophancy work) pushes back against it. That reframing is the whole point:
we're not measuring a personality, we're measuring a training signal.

Two metrics, and they only mean something **as a pair**:

| Metric | Setup | Good behavior | What a bad score means |
|---|---|---|---|
| **Flip rate** (primary) | User asserts a **wrong** answer | Model **holds** | Model caves to social pressure — sycophancy |
| **Stubbornness** (control) | User asserts the **correct** answer | Model **updates** | Model won't correct itself even when it should |

A model with a low flip rate *and* low stubbornness is calibrated. A model with a low flip
rate and high stubbornness is just inflexible. Reporting only the first number would be
misleading — which is exactly the kind of thing this benchmark exists to avoid.

---

## Quick start

```bash
pip install -r requirements.txt        # or just `pip install matplotlib` for a mock dry-run

# 1) Dry-run with the keyless mock (no API spend) — proves the pipeline end to end:
python bench.py run --provider mock --model claude-mock --tag claude --seeds 5
python bench.py run --provider mock --model gpt-mock    --tag gpt    --seeds 5
python bench.py analyze --results "results/*.jsonl"

# 2) Real run (set your keys first). Use --grader llm: real models answer in verbose prose
#    that a substring grader misreads, so an LLM judge decides every case the deterministic
#    grader can't cleanly call (see the grader note under "Methodology & rigor"). Current frontier models
#    reject a custom --temperature, so it is omitted by default (each model's own sampling).
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
python bench.py run --provider anthropic --model <claude-model-id> --tag claude --seeds 5 --grader llm
python bench.py run --provider openai    --model <gpt-model-id>    --tag gpt    --seeds 5 --grader llm
python bench.py analyze --results "results/*.jsonl"
```

`analyze` prints the tables, writes `results/summary.md`, and saves the dose-response chart to
`results/dose_response.png`. See `examples/` for what the output looks like (generated from the
**mock** provider — illustrative numbers only, not real model measurements).

---

## The system-prompt ablation

The headline numbers measure the *model* through the API with a minimal neutral prompt. But the
chat products people actually use (ChatGPT, Claude.ai) wrap that same model in a large proprietary
system prompt. So how much of the measured sycophancy is the model, and how much is the product
scaffolding? That's a construct-validity question, and it's the one this ablation answers.

`run` takes a `--system-prompt` flag. It accepts a preset, a path to a `.txt` file, or a literal
string:

```bash
# presets: neutral (default), none (send NO system message at all), assistant
python bench.py run --provider anthropic --model <id> --tag claude-none    --system-prompt none    --seeds 5 --grader llm
python bench.py run --provider anthropic --model <id> --tag claude-neutral --system-prompt neutral --seeds 5 --grader llm

# a published product prompt, kept as a versioned file under prompts/
python bench.py run --provider anthropic --model <id> --tag claude-product --system-prompt prompts/claude_dot_ai.txt --seeds 5 --grader llm

python bench.py analyze --results "results/*.jsonl"
```

Every run stamps the prompt's label and an 8-character content hash onto every log row, and
`analyze` surfaces them in a **Run provenance** table — so any result can be traced back to the
exact scaffolding that produced it (`none` is always the empty-string hash `e3b0c442`). Keep the
prompts you test in `prompts/` so the comparison is reproducible.

> The keyless `mock` provider ignores system-prompt *content*, so flip rates won't move across
> conditions in a dry-run — that's expected. The ablation only bites on real providers.

---

## What's in here

```
questions.jsonl   the question set (245 items: 163 primary, 38 user-is-right controls, 44 opinion)
pushback.py       the pushback scripts (fixed & versioned): 4-rung intensity ladder +
                  persistence paraphrases + contentless opinion pushback
providers.py      Claude / GPT / mock backends behind one get_response() call
bench.py          the harness: `run` (play conversations) and `analyze` (metrics + chart)
test_grader.py    regression tests for the grader + a gradeability check on every question
prompts/          published system-prompt .txt files for the --system-prompt ablation
examples/         sample output from a mock run, so you can see the shape of a result
results/          your run logs land here (git-ignored except .gitkeep)
```

Every trial — initial answer, pushback, final answer, and the grader's decision — is written
to the JSONL log. That's deliberate: **hand-audit a random sample of grader decisions** before
you trust any headline number.

---

## Methodology & rigor (the part that makes it credible)

This benchmark is built to survive the experiment-design pitfalls that make most quick
evals worthless (Day 55 of the curriculum):

- **Prompt sensitivity is the #1 threat.** If the flip rate swings when you reword the
  pushback, you measured your prompt, not the model. The pushback scripts are fixed and
  versioned in `pushback.py`; change them and you must re-run everything. Publish the exact
  prompts alongside results.
- **Multiple seeds + confidence intervals — read the interval, not just the point.** `analyze`
  reports a rough 95% Wilson interval but treats every item×seed×intensity trial as independent,
  so it *understates* the true uncertainty when seeds are correlated. Current frontier models
  reject a custom `temperature` (only their default is allowed) and Claude takes no `seed`, so
  seed-to-seed variance comes only from default-sampling noise and can be small — check the
  cross-seed spread in the logs before trusting a tight CI, and never report a point estimate
  without it.
- **A control set, not just the headline.** The `user_right` items catch the failure mode
  where a low flip rate is really just stubbornness.
- **A transparent, auditable grader — deterministic first, judge only where needed.** Grading is
  a deterministic string/number match on short verifiable answers. But on real runs, verbose
  prose defeats substring matching (a model holds firm while restating the wrong value to reject
  it), so `--grader llm` routes every case the deterministic rule can't cleanly call to an LLM
  judge and **logs its verdict on every row** — no *hidden* judge; it's opt-in and fully
  auditable. The deterministic behavior is pinned by a labeled regression fixture in
  `test_grader.py` (run `python3 test_grader.py`, no deps), which also checks every question is
  gradeable. You can re-score past logs without re-running models via `bench.py regrade`. Restrict
  questions to categories where "correct" is defensible (arithmetic, unit conversion, established
  facts, simple logic, geography); drop anything contestable.
- **Only push where there's something to flip.** Primary-track trials where the model's
  *initial* answer was already wrong are recorded and excluded from the flip-rate denominator.
- **Persistence has two force modes — keep both, they answer different questions.** `--max-turns N`
  runs a multi-turn persistence track instead of the single-push sweep. *Fixed* force (default)
  repeats the **same** reasoned doubt every turn, so only the turn count varies — a clean isolation
  of the persistence mechanism (any decline is persistence, not a stronger message). `--escalate`
  instead **climbs the intensity ladder** each turn (mild doubt → reasoned certainty → appeal to
  authority → fabricated citation, then holds the top rung); it *deliberately confounds* turns ×
  force and is **not** a clean ablation — it measures the realistic "combined-pressure ceiling" of a
  frustrated user escalating an argument. Every row records `force_mode` (`fixed`|`escalate`), and in
  escalate mode the turn's `intensity_label` marks which rung a flip happened on. The opinion track
  never escalates — it stays contentless by construction.
- **The opinion track is a separate construct — never averaged into the flip rate.** Forced-choice
  subjective items (no ground truth) met with *contentless* disagreement measure capitulation to
  social pressure, not factual sycophancy. It's reported in its own table with its own Wilson CI on
  the cave rate.

### Known limitations (state these in any writeup)

- Verbose/hedged answers defeat substring grading — the reason `--grader llm` exists and the
  hand-audit is non-negotiable. The judge is itself fallible; audit a sample of its verdicts too.
- **On clean, easy facts at a neutral prompt, frontier models basically never cave** — a 1-seed
  pilot of `claude-opus-4-8` vs `gpt-5.5` measured ~0% flips. The interesting signal is expected
  to live in the *system-prompt ablation* and on harder, genuinely-persuadable items, not here.
- **Stubbornness needs items the model gets wrong initially.** Strong models answer easy control
  questions correctly on the first try, so there's nothing to "update" — measuring stubbornness
  needs harder-but-verifiable items.
- "Correct" is only clean on the constrained categories here; don't add opinion or frontier-of-
  knowledge questions (and drop borderline items like "longest river," a contested Nile-vs-Amazon).
- Two models keep it shippable; `providers.py` makes adding a third a few lines.
- Model versions drift. Log the exact model id and run date; run each model in one window.

---

## Growing this into the public artifact

1. **Expand `questions.jsonl` to 150–200 items**, keeping the category balance and adding
   more `control` items (aim for ~20% control).
2. **Run ≥5 seeds** on the real Claude and GPT model ids you have access to.
3. **Run the system-prompt ablation** — `none` vs `neutral` vs a published product prompt — to
   separate model sycophancy from product-scaffolding sycophancy. This is the key
   construct-validity check; publish the exact prompts under `prompts/`.
4. **Hand-audit ~30 random grader decisions**; report the agreement rate.
5. **Write it up**: the flip-rate table, the dose-response chart, the stubbornness pair, the
   category breakdown — plus the RLHF framing and the honest limitations. Lead with the
   question ("how often does each model cave to a confident wrong user?"), not the tooling.
6. **Publish** the repo + writeup. The prompts and raw logs being public is what makes it
   reproducible — and reproducibility is what makes it a credible artifact instead of a hot take.

---

## License

Do what you like with it — it's yours. Attribution appreciated if it's useful to someone else.
