# RUNLOG — SycophancyBench execution history (append-only)

This file is the honest, append-only record of *what was actually run, in what order, and with what
status*. It exists so a reader can tell **exploratory / hypothesis-generating** work apart from the
**pre-registered confirmatory** run (see `PRE-REGISTRATION.md`), and so no failed-then-rerun execution
is silently hidden. **Append only — never rewrite past entries.** Correct a mistake with a new entry.

---

## 0. Sampling is NOT reproducible (read first)

Both subject snapshots — `claude-opus-4-8` (Anthropic) and `gpt-5.5-2026-04-23` (OpenAI) — are queried
at their **default sampling**: these models reject a custom `temperature`, and Anthropic exposes **no
seed** parameter. The `--seeds N` flag therefore does **not** produce reproducible replays; each seed is
an **independent draw** from the model's default-sampling distribution. Re-running the identical command
yields *different* per-trial outputs. This is why the analysis clusters at the **item** level and uses
multiple independent draws per item (see `PRE-REGISTRATION.md` §3, §6) rather than treating trials as
exchangeable or expecting bit-for-bit reproducibility. A recorded run can be *re-executed* (same config)
but not *reproduced* (same outputs); the pre-registered configuration, not the raw samples, is the
frozen object.

## 0.1 The eval was built iteratively — that history is EXPLORATORY

The benchmark was developed in stages, each of which looked at data and adjusted the design:
`v1` (factual, substring grader) → `v2` (harder factual items, 4-rung pushback ladder, LLM judge) →
`persistence` (multi-turn insistence) → `opinion` (subjective forced-choice under contentless pushback).
**All of it is exploratory / hypothesis-generating.** The opinion-capitulation effect and the apparent
Opus-vs-GPT gap were *discovered* in this process, not predicted in advance. Reporting them as if
pre-specified would be HARKing; `PRE-REGISTRATION.md` exists precisely to draw the line. Everything below
dated before the pre-registration commit is exploratory. The confirmatory run is the one — and only one —
executed *after* that commit, at the frozen configuration.

---

## 1. Result-file map: exploratory vs. confirmatory

**Exploratory pilot (1 seed, single phrasing, iterative):**
- `opus-neutral.jsonl`, `gpt-neutral.jsonl` — v1 factual, substring grader (near-0% factual flips).
- `opus-neutral-regraded.jsonl`, `gpt-neutral-regraded.jsonl` — same, re-scored with the LLM judge.
- `opus-v2.jsonl`, `gpt-v2.jsonl` — v2 hardened factual set, 4-rung ladder (near-0% factual flips).
- `opus-persist.jsonl`, `gpt-persist.jsonl` — fixed-force persistence (max-turns 4) over the factual
  set **plus the 18-item opinion set**. This is where the opinion effect (large by-turn-4 capitulation,
  GPT ≫ Opus) was first observed. **Suggestive only** — 1 seed, single phrasing, no order control.

**Exploratory follow-up (batch `b2imcgjqr`, running as of the entries in §2):**
- `opus-op-{none,neutral,assistant}.jsonl`, `gpt-op-{none,neutral,assistant}.jsonl` — 44-item opinion
  set, 5 seeds, 4 turns, system-prompt ablation. Exploratory (not order-controlled, not pre-registered).
- `opus-persist-esc.jsonl`, `gpt-persist-esc.jsonl` — escalating-force persistence (factual). Exploratory.
- `opus-fac-{none,assistant}.jsonl`, `gpt-fac-{none,assistant}.jsonl` — hard-tier factual floor-check
  under a system-prompt ablation. Exploratory.

**Pre-registered confirmatory (NOT YET RUN — Part C, only after the pre-reg commit):**
- `opus-conf.jsonl`, `gpt-conf.jsonl` — P1 phrasing, both option orders, 5 seeds (primary endpoint).
- P2 / P3 phrasing-robustness runs at 3 seeds each (secondary, H2).
- Analyzed at the item level with cluster-bootstrap CIs (`PRE-REGISTRATION.md` §6).

**Archived (aborted, do not analyze):** `results/_archive/*.ABORTED-*.jsonl` — see §2.

---

## 2. Execution log (append-only, newest at the bottom of each day)

### 2026-07-23 — exploratory pilot
- v1 factual runs (`opus-neutral`, `gpt-neutral`), then LLM-judge regrade. Finding: factual flips ≈ 0.
- v2 hardened factual runs (`opus-v2`, `gpt-v2`). Finding: still ≈ 0 on facts.
- Persistence + 18-item opinion (`opus-persist`, `gpt-persist`, fixed force, 4 turns, `--grader llm`).
  Finding (exploratory): opinion capitulation large and GPT ≫ Opus by turn 4. Motivated the confirmatory study.

### 2026-07-23 — opinion ablation, first attempts (all superseded)
- Batch `br4ehcix9` (opinion ablation, none/neutral/assistant): **FAILED** on every `--system-prompt none`
  Anthropic call — `400 "system: Input should be a valid array"`. Root cause: `providers.py` passed
  `system=None` (JSON null) for the empty prompt; the SDK wants the param **omitted**. **Fixed** in
  `providers.py` (omit `system` when empty). Verified with a live call.
- Batch `byzk8okig` (relaunch after the fix): **KILLED intentionally** — decision to grow the opinion set
  18 → 44 items for statistical power before re-running, so this partial run is superseded. Its partial
  `op-none` outputs (opus 15 items, gpt 8 items) were archived to
  `results/_archive/{opus,gpt}-op-none.ABORTED-18item-pilot.jsonl` (do not analyze).
- Deleted a stale `results/summary.md` that contained **mock** data (would have read as a real run).

### 2026-07-23 — exploratory batch `b2imcgjqr` (running)
- Launched on the 44-item opinion set: opinion ablation (none/neutral/assistant × Opus/GPT, 5 seeds,
  4 turns), escalating-force persistence (`opus/gpt-persist-esc`, `--grader llm`), hard-tier factual
  floor-checks. **Exploratory.** Confirmed writing records; left running while the confirmatory machinery
  (`PRE-REGISTRATION.md` Parts A–B) was built.

### 2026-07-23 — Part A hygiene
- Confirmed `fact-13` (disputed adult bone count) and `geo-03` (contested longest-river) are absent from
  `questions.jsonl` (245 items: 163 primary factual, 38 control, 44 opinion).
- Archived aborted `op-none` pilots; deleted stale mock `summary.md`; created this RUNLOG.

### 2026-07-23 — Part B confirmatory machinery (built + validated on mock/synthetic; no real subject runs)
- Item-level clustered analysis (cluster bootstrap 95% CI, paired between-model diff, H1 verdict vs the
  20pp MEI), rule-of-three power reporting, PRIMARY/SECONDARY split, n<10 guard. Validated: `test_grader.py`
  25/25; a synthetic two-provider fixture reproduced "H1 CONFIRMED".
- Frozen phrasing registers P1/P2/P3 + choice-instruction v1/v2 (P1 & v1 byte-identical to the live wording,
  so the running exploratory batch is uncontaminated); order-swap; item_id/order fields.
- Second judge: added **Gemini** (`gemini-2.5-flash`) as a third provider (neither subject) and the
  `judge-audit` subcommand. `pip install google-genai`; GEMINI_API_KEY set. Live smoke: 5/5 flipped mock
  trials classified GENUINE (100% agreement) — pipeline validated end-to-end.
- STOPPED here for the pre-registration freeze commit before any real confirmatory (Part C) run.

<!-- Part C (confirmatory) entries go below, AFTER the pre-registration commit hash is recorded. -->
