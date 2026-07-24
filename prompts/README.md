# System prompts

Home for **published** system-prompt `.txt` files used by the `--system-prompt` ablation.

The headline benchmark measures the *model* through the API with a minimal neutral prompt. But
the chat products people actually use (ChatGPT, Claude.ai) wrap that same model in a large
proprietary system prompt. Swapping the system prompt quantifies how much of the measured
sycophancy is the model vs. the product scaffolding — the key construct-validity check.

## Usage

`--system-prompt` accepts a built-in preset, a path to a `.txt` file in here, or a literal string:

```bash
# presets
python bench.py run --provider anthropic --model <id> --tag claude-none    --system-prompt none    --seeds 5
python bench.py run --provider anthropic --model <id> --tag claude-neutral --system-prompt neutral --seeds 5

# a published product prompt kept as a versioned file
python bench.py run --provider anthropic --model <id> --tag claude-product --system-prompt prompts/claude_dot_ai.txt --seeds 5
```

Built-in presets: `neutral` (the API-default used for the headline numbers), `none` (no system
message at all), `assistant` ("helpful, harmless, and honest").

## Why keep them as files

The ablation is only meaningful if the exact prompt is public and versioned alongside the results.
Every run stamps the prompt's label and an 8-character content hash into the logs, and `analyze`
surfaces them in a **Run provenance** table — so any result can be traced back to the precise
scaffolding that produced it.

Only commit prompts that are genuinely public (officially published or already-leaked product
prompts). Don't paste anything under NDA.
