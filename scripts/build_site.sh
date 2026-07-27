#!/usr/bin/env bash
# Build the GitHub Pages site (docs/) from the canonical sources:
#   docs/writeup.html   <- rendered from WRITEUP.md (pandoc, gfm pipe tables)
#   docs/dashboard.html <- copied from results/dashboard.html
# docs/index.html is a static hand-written landing page and is NOT touched here.
# Run from the repo root: scripts/build_site.sh   (then commit docs/ in the same commit
# as any WRITEUP.md edit, so the page never drifts from the canonical markdown).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v pandoc >/dev/null || { echo "pandoc not found (brew install pandoc)"; exit 1; }
[ -f WRITEUP.md ] || { echo "WRITEUP.md not found"; exit 1; }
[ -f results/dashboard.html ] || { echo "results/dashboard.html not found"; exit 1; }

HASH=$(git log -1 --format=%h -- WRITEUP.md)
DATE=$(git log -1 --format=%cs -- WRITEUP.md)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/head.html" <<'EOF'
<meta property="og:type" content="article">
<meta property="og:title" content="Do Frontier Models Cave? A Reproducible, Pre-Registered Benchmark for Sycophancy Under User Pushback">
<meta property="og:description" content="On facts, Claude Opus and GPT-5.5 almost never cave, even against a fabricated citation; on subjective picks GPT abandons its stance on 65.9% of items vs 33.9% for Opus.">
<style>
:root{color-scheme:light dark}
body{max-width:44rem;margin:0 auto;padding:2rem 1.25rem 4rem;
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:#1a1f27;background:#ffffff}
h1{font-size:1.6rem;line-height:1.25}
h2{font-size:1.25rem;margin-top:2.2em;border-bottom:1px solid #d8dee6;padding-bottom:.25em}
h3{font-size:1.05rem;margin-top:1.8em}
a{color:#0b6e8c}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.9em;
  background:#eef1f5;border-radius:4px;padding:.08em .3em}
pre{background:#eef1f5;border-radius:6px;padding:.8em;overflow-x:auto}
pre code{background:none;padding:0}
table{border-collapse:collapse;display:block;overflow-x:auto;margin:1em 0;font-size:.95em}
th,td{border:1px solid #d8dee6;padding:.4em .7em;text-align:left}
th{background:#f3f5f8}
blockquote{border-left:3px solid #c6ced8;margin-left:0;padding-left:1em;color:#4a5462}
hr{border:0;border-top:1px solid #d8dee6;margin:2.5em 0}
footer{margin-top:3.5em;padding-top:1em;border-top:1px solid #d8dee6;font-size:.85em;color:#68727f}
@media (prefers-color-scheme:dark){
  body{color:#e2e7ee;background:#10141a}
  h2{border-color:#2a323d}
  a{color:#57b8d6}
  code,pre{background:#1b2129}
  th,td{border-color:#2a323d}
  th{background:#181e26}
  blockquote{border-color:#3a4450;color:#a6b0bd}
  hr,footer{border-color:#2a323d}
  footer{color:#8b95a2}
}
</style>
EOF

cat > "$TMP/foot.html" <<EOF
<footer>Generated from <a href="https://github.com/celomedrado/SycophancyBench/blob/main/WRITEUP.md">WRITEUP.md</a> (canonical) at commit <code>${HASH}</code>, ${DATE}.</footer>
EOF

pandoc WRITEUP.md -f gfm -t html5 -s \
  --metadata pagetitle="Do Frontier Models Cave? A Reproducible, Pre-Registered Benchmark for Sycophancy Under User Pushback" \
  -H "$TMP/head.html" -A "$TMP/foot.html" \
  -o docs/writeup.html

cp results/dashboard.html docs/dashboard.html
echo "built docs/writeup.html (from WRITEUP.md @ ${HASH}) and docs/dashboard.html"
