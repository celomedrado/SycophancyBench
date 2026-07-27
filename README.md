# SycophancyBench

SycophancyBench measures what happens when a user pushes back with no argument at all:
on factual questions, `claude-opus-4-8` and `gpt-5.5` almost never cave, even against a
fabricated citation, while on subjective picks GPT abandons its stance on 65.9% of items
versus 33.9% for Opus.

## Headline (pre-registered primary endpoint)

Item level, n = 44 subjective items, cluster-bootstrap 95% CI:

| Model | Capitulation under contentless disagreement |
|---|---|
| `claude-opus-4-8` | 33.9% [25.0, 43.0] |
| `gpt-5.5-2026-04-23` | 65.9% [55.0, 76.1] |
| **Gap (GPT minus Opus)** | **+32.0 [21.8, 41.6]**, entirely above the pre-registered 20-point minimum effect |

## Three receipts

- **Controls.** With a neutral or agreeing user on the same items, measured drift is 0%
  for both models. The movement is caused by the disagreement itself
  ([control-summary](results/control-summary.md)).
- **Facts.** Opus: 0 flips in 489 escalating-force trials ending at a fabricated
  citation. GPT: Hold@4 = 98.8%.
- **Honesty.** The mild-register GPT cell is reported as an instrument failure, not a
  finding: GPT stops emitting a parseable choice on 77.4% of turns there, and re-scoring
  the untagged turns can flip the sign, so no verdict is claimed in either direction.

## Why you can trust it

- The pre-registration (hypotheses, frozen stimuli, item-level analysis, 20-point
  minimum effect, falsification rule) was committed at
  [`413f44d`](PRE-REGISTRATION.md) BEFORE any confirmatory data was collected. The
  control-arm hypotheses (section H4) were likewise frozen at `ad8004a`, and the batch
  log that collected their data opens 32 seconds after that commit and stamps its hash
  (`results/_batch_h4.log`).
- Every deviation is logged append-only in PRE-REGISTRATION.md section 11, including
  one retracted claim (the P3 "reversal") and two post-hoc parser fixes found by
  external review.
- An independent third-provider judge (Gemini, neither subject model) rates 94% (Opus)
  and 86% (GPT) of the deterministic flips as genuine reversals; under that stricter
  reading the gap is about 25 points, still above threshold.
- Every number regenerates from the raw logs in `results/` (about 55k API calls).

## Read the evidence

- [WRITEUP.md](WRITEUP.md), the full paper-style report
- [PRE-REGISTRATION.md](PRE-REGISTRATION.md), the frozen design and deviations log
- [RUNLOG.md](RUNLOG.md), the append-only execution history, failures included
- [Live dashboard](https://celomedrado.github.io/SycophancyBench/), every conversation,
  grade, and judge verdict, browsable
- [`results/`](results/), the raw per-trial logs

## Reproduce

```bash
pip install -r requirements.txt
```

Put keys in `.env` or the environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for the judge.

```bash
python3 bench.py run --provider anthropic --model claude-opus-4-8 --tag my-run \
  --categories opinion --swap-options --seeds 5 --max-turns 4
python3 bench.py analyze --results "results/my-run.jsonl"
python3 test_grader.py   # zero-dependency test suite, 34 passing
```

The keyless dry run works without any API key: `--provider mock --model gpt-mock`.

## Scope

This benchmark measures stated-preference lability on low-stakes forced choices under
contentless social pressure, for two models from one lab pairing. It is not a universal
sycophancy score. The factual track is exploratory and reported as an upper bound.

## License and data note

MIT (see [LICENSE](LICENSE)). The raw logs contain verbatim model outputs from the
Anthropic, OpenAI, and Google APIs, published for research and evaluation purposes.
