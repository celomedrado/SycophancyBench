# Quarantined result logs — auditable, NOT analyzable

Files here are kept for provenance but must never be plotted, summarized, or cited. Each name
states why it was quarantined.

- `opus-persist-esc.INVALID-judge-outage-and-pre-fix-grader.jsonl` — escalating-force persistence,
  Opus, 1 seed. Invalid for two independent reasons: (a) an OpenAI judge quota outage turned
  267/354 rows into judge failures that the old code recorded as `ambiguous`, silently deleting
  ~75% of the sample (and the loss rises monotonically by turn, so it biases the dose-response as
  well as shrinking n); (b) it was graded before the negation-polarity fix, so 12 of its rows are
  false flips. The GPT arm (`gpt-persist-esc`) never produced a single row, so the comparison was
  single-arm anyway. Superseded by a re-run at >=3 seeds per model with the third-provider judge
  and the `error`-outcome guard enabled.
- `gpt-conf-p3.INCOMPLETE-openai-quota-15of44.jsonl` — superseded by a complete 44/44 re-run.
- `gpt-conf.INCOMPLETE-openai-quota-28of44.jsonl` — superseded by a complete 44/44 re-run.
- `opus-conf-p2.INCOMPLETE-ratelimit-16of44.jsonl` — superseded by a complete 44/44 re-run.
- `{opus,gpt}-op-none.ABORTED-18item-pilot.jsonl` — 18-item opinion pilot, superseded by the
  44-item set.
