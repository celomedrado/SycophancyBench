#!/usr/bin/env python3
"""Build a self-contained HTML dashboard from SycophancyBench result logs.

    python make_dashboard.py --results "results/*-regraded.jsonl" --out results/dashboard.html

Reads one or more result .jsonl logs, embeds every trial (question, the model's initial
and final answers, the rendered pushback, the deterministic grade, and the LLM judge's
verdict + reasoning) into a single static HTML file, and computes the headline metrics in
the browser so filters recompute live. No external assets. Opens offline, shares as one file.
"""

import argparse
import glob
import html
import json
import sys

try:
    from pushback import (render as render_pushback, render_persist, render_opinion,
                          INTENSITY_LABELS)
except Exception:  # keep the generator usable even if pushback.py moves
    INTENSITY_LABELS = ["mild_doubt", "confident_assertion", "appeal_to_authority"]
    def render_pushback(i, claim):
        return f"[pushback {i} - claim: {claim}]"
    def render_persist(t, claim):
        return f"[persist turn {t} - claim: {claim}]"
    def render_opinion(t, other):
        return f"[opinion turn {t} - other: {other}]"


def load(patterns):
    rows = []
    for pat in patterns:
        for fp in glob.glob(pat):
            with open(fp) as f:
                rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _pushback_text(r):
    """Reconstruct the exact user pushback shown to the model for this row, matching the
    machinery bench.py used: opinion -> render_opinion, multi-turn persist -> render_persist,
    single-push factual -> render (intensity ladder)."""
    direction = r.get("direction", "user_wrong")
    turn = r.get("turn", 1) or 1
    try:
        if direction == "opinion":
            if r.get("initial_choice") is None:
                return ""  # never committed; no pushback was run
            other = r.get("option_b") if r.get("pushed_choice") == "B" else r.get("option_a")
            return render_opinion(turn - 1, other or "the other option")
        if r.get("mode") == "persist":
            if r.get("force_mode") == "escalate":   # climbed the intensity ladder, not paraphrases
                rung = min(turn - 1, len(INTENSITY_LABELS) - 1)
                return render_pushback(rung, r.get("user_claim", ""))
            return render_persist(turn - 1, r.get("user_claim", ""))
        intensity = r.get("intensity")
        if intensity is not None:
            return render_pushback(intensity, r.get("user_claim", ""))
    except Exception:
        return ""
    return ""


def compact(r):
    """Pull just the fields the dashboard renders, and reconstruct the pushback text."""
    intensity = r.get("intensity")
    pushback = _pushback_text(r) if r.get("outcome") not in ("skipped_initial_wrong",
                                                             "skipped_no_initial_commit") else ""
    fj = r.get("final_judge") or None
    ij = r.get("initial_judge") or None
    return {
        "tag": r.get("tag", "?"),
        "model": r.get("model", "?"),
        "id": r.get("id", "?"),
        "category": r.get("category", "?"),
        "difficulty": r.get("difficulty", "standard"),
        "direction": r.get("direction", "user_wrong"),
        "seed": r.get("seed"),
        "intensity": intensity,
        "intensity_label": r.get("intensity_label"),
        "turn": r.get("turn", 1),
        "mode": r.get("mode", "single"),
        "force_mode": r.get("force_mode", "fixed"),
        "question": r.get("question", ""),
        "correct": r.get("correct_answer", ""),
        "claim": r.get("user_claim", ""),
        "common_wrong": r.get("common_wrong", ""),
        "pushback": pushback,
        "initial_answer": r.get("initial_answer", ""),
        "initial_grade": r.get("initial_grade", ""),
        "final_answer": r.get("final_answer", ""),
        "final_grade": r.get("final_grade", ""),
        "outcome": r.get("outcome", ""),
        # opinion-track fields (present only on direction=="opinion" rows)
        "option_a": r.get("option_a", ""),
        "option_b": r.get("option_b", ""),
        "initial_choice": r.get("initial_choice"),
        "final_choice": r.get("final_choice"),
        "pushed_choice": r.get("pushed_choice"),
        "sys_label": r.get("system_prompt_label", "?"),
        "sys_hash": r.get("system_prompt_hash", "?"),
        # judge audit trail (the piece humans most want to inspect)
        "judge": ({"det": fj.get("deterministic"), "verdict": fj.get("verdict"),
                   "model": fj.get("judge_model"), "raw": fj.get("raw", "")} if fj else None),
        "init_judge": ({"det": ij.get("deterministic"), "verdict": ij.get("verdict"),
                        "raw": ij.get("raw", "")} if ij else None),
    }


def compact_q(q):
    """Fields the Question-set tab shows."""
    return {
        "id": q.get("id", ""), "category": q.get("category", ""),
        "direction": q.get("direction", "user_wrong"),
        "difficulty": q.get("difficulty", "standard"),
        "question": q.get("question", ""),
        "correct": q.get("correct_answer", ""),
        "claim": q.get("user_claim", ""),
        "common_wrong": q.get("common_wrong", ""),
        "option_a": q.get("option_a", ""),
        "option_b": q.get("option_b", ""),
    }


def build(rows, questions, generated_note):
    data = [compact(r) for r in rows]
    # ensure_ascii=True so the embedded data is pure ASCII (\uXXXX escapes) and renders
    # correctly no matter what charset the host serves the page with.
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    qpayload = json.dumps([compact_q(q) for q in questions], ensure_ascii=True, separators=(",", ":"))
    return HTML_TEMPLATE.replace("/*__DATA__*/null", payload) \
                        .replace("/*__QUESTIONS__*/null", qpayload) \
                        .replace("/*__LABELS__*/[]", json.dumps(INTENSITY_LABELS)) \
                        .replace("__GENERATED__", html.escape(generated_note))


def main():
    p = argparse.ArgumentParser(description="Build the SycophancyBench results dashboard")
    p.add_argument("--results", nargs="+",
                   default=["results/*-regraded.jsonl"],
                   help="result .jsonl files or globs (default: results/*-regraded.jsonl)")
    p.add_argument("--out", default="results/dashboard.html")
    p.add_argument("--questions", default="questions.jsonl",
                   help="question set shown in the Questions tab (default: questions.jsonl)")
    p.add_argument("--note", default="", help="optional provenance note shown in the header")
    args = p.parse_args()

    rows = load(args.results)
    if not rows:
        print(f"[dashboard] no records matched {args.results}", file=sys.stderr)
        sys.exit(1)
    questions = load([args.questions]) if args.questions else []
    html_out = build(rows, questions, args.note or f"{len(rows)} trials from {', '.join(args.results)}")
    with open(args.out, "w") as f:
        f.write(html_out)
    print(f"[dashboard] wrote {args.out} ({len(rows)} trials, {len(questions)} questions, {len(html_out)//1024} KB)")


# --------------------------------------------------------------------------
# The page. Data is injected in place of the /*__DATA__*/null token.
# --------------------------------------------------------------------------
HTML_TEMPLATE = r"""<style>
:root{
  --bg:#F4F6F8; --surface:#FFFFFF; --surface-2:#EEF1F5; --border:#E1E6EC;
  --text:#141922; --muted:#586274; --faint:#8A93A2;
  --accent:#0C8DA1; --accent-2:#0A7183;
  --held:#1C9C6A; --flip:#D63B5E; --amb:#B4801C; --neutral:#7A8698;
  --held-b:rgba(28,156,106,.13); --flip-b:rgba(214,59,94,.13); --amb-b:rgba(180,128,28,.14); --neutral-b:rgba(122,134,152,.14);
  --shadow:0 1px 2px rgba(16,22,30,.05),0 4px 16px rgba(16,22,30,.06);
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --r:10px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0D1116; --surface:#161B22; --surface-2:#1B212B; --border:#262E3A;
    --text:#E6EBF2; --muted:#97A1B2; --faint:#5A6472;
    --accent:#2CBAD0; --accent-2:#5AD1E3;
    --held:#39BC82; --flip:#E85477; --amb:#DCA53F; --neutral:#7B8698;
    --held-b:rgba(57,188,130,.15); --flip-b:rgba(232,84,119,.15); --amb-b:rgba(220,165,63,.15); --neutral-b:rgba(123,134,152,.16);
    --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.28);
  }
}
:root[data-theme="light"]{
  --bg:#F4F6F8; --surface:#FFFFFF; --surface-2:#EEF1F5; --border:#E1E6EC;
  --text:#141922; --muted:#586274; --faint:#8A93A2;
  --accent:#0C8DA1; --accent-2:#0A7183;
  --held:#1C9C6A; --flip:#D63B5E; --amb:#B4801C; --neutral:#7A8698;
  --held-b:rgba(28,156,106,.13); --flip-b:rgba(214,59,94,.13); --amb-b:rgba(180,128,28,.14); --neutral-b:rgba(122,134,152,.14);
  --shadow:0 1px 2px rgba(16,22,30,.05),0 4px 16px rgba(16,22,30,.06);
}
:root[data-theme="dark"]{
  --bg:#0D1116; --surface:#161B22; --surface-2:#1B212B; --border:#262E3A;
  --text:#E6EBF2; --muted:#97A1B2; --faint:#5A6472;
  --accent:#2CBAD0; --accent-2:#5AD1E3;
  --held:#39BC82; --flip:#E85477; --amb:#DCA53F; --neutral:#7B8698;
  --held-b:rgba(57,188,130,.15); --flip-b:rgba(232,84,119,.15); --amb-b:rgba(220,165,63,.15); --neutral-b:rgba(123,134,152,.16);
  --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px 80px}
a{color:var(--accent)}
h1,h2,h3{text-wrap:balance;margin:0}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}

/* header */
header{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
.hbar{max-width:1120px;margin:0 auto;padding:14px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.brand{display:flex;flex-direction:column;gap:2px;margin-right:auto}
.brand h1{font-size:19px;font-weight:640;letter-spacing:-.01em}
.brand .sub{font-size:12.5px;color:var(--muted)}
.brand .sub b{color:var(--text);font-weight:620}
.iconbtn{appearance:none;border:1px solid var(--border);background:var(--surface);color:var(--muted);
  width:36px;height:36px;border-radius:9px;cursor:pointer;font-size:15px;display:grid;place-items:center}
.iconbtn:hover{color:var(--text);border-color:var(--accent)}
.iconbtn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* provenance chips */
.prov{display:flex;gap:8px;flex-wrap:wrap;padding:16px 0 4px}
.chip{display:inline-flex;align-items:center;gap:8px;background:var(--surface);border:1px solid var(--border);
  border-radius:999px;padding:5px 12px 5px 8px;font-size:12px;box-shadow:var(--shadow)}
.chip .dot{width:9px;height:9px;border-radius:3px}
.chip b{font-weight:620}
.chip .mono{color:var(--muted);font-size:11px}

/* section */
section{margin-top:30px}
.sechead{display:flex;align-items:baseline;gap:12px;margin-bottom:14px}
.sechead h2{font-size:15px;font-weight:620;letter-spacing:-.005em}
.sechead .note{font-size:12.5px;color:var(--muted)}

/* summary cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:18px 18px 16px;box-shadow:var(--shadow)}
.card .top{display:flex;align-items:center;gap:9px;margin-bottom:12px}
.card .top .dot{width:11px;height:11px;border-radius:3px}
.card .top .name{font-family:var(--mono);font-size:13px;font-weight:600}
.card .top .cover{margin-left:auto;font-size:11px;color:var(--faint)}
.big{display:flex;align-items:baseline;gap:10px}
.big .rate{font-size:40px;font-weight:680;letter-spacing:-.02em;line-height:1}
.big .ci{font-size:13px;color:var(--muted)}
.big .lbl{font-size:12px;color:var(--muted);margin-left:auto;text-align:right}
.stack{display:flex;height:9px;border-radius:5px;overflow:hidden;margin:14px 0 9px;background:var(--surface-2)}
.stack span{height:100%}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:baseline}
.subrow{display:flex;gap:18px;margin-top:13px;padding-top:13px;border-top:1px solid var(--border);font-size:12.5px;color:var(--muted)}
.subrow div b{color:var(--text);font-weight:620;font-family:var(--mono)}

/* charts */
.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.panels{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;box-shadow:var(--shadow)}
.panel h3{font-size:13px;font-weight:600;margin-bottom:4px}
.panel .cap{font-size:12px;color:var(--muted);margin-bottom:10px}
.catrow{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:12.5px}
.catrow .clabel{width:120px;color:var(--muted);flex:none}
.catbar{flex:1;height:8px;background:var(--surface-2);border-radius:4px;position:relative;overflow:hidden}
.catbar i{position:absolute;left:0;top:0;height:100%;border-radius:4px}
.catrow .cval{width:74px;text-align:right;flex:none;font-family:var(--mono);color:var(--muted)}

/* explorer */
.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.filters input[type=search]{flex:1;min-width:180px;background:var(--surface);border:1px solid var(--border);
  border-radius:9px;padding:9px 12px;color:var(--text);font-family:var(--sans);font-size:13px}
.filters input:focus-visible{outline:2px solid var(--accent);outline-offset:1px;border-color:var(--accent)}
.seg{display:flex;background:var(--surface);border:1px solid var(--border);border-radius:9px;overflow:hidden}
.seg button{appearance:none;border:0;background:transparent;color:var(--muted);padding:8px 12px;font-size:12.5px;
  font-family:var(--sans);cursor:pointer;border-right:1px solid var(--border)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed=true]{background:var(--accent);color:#fff}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.count{font-size:12.5px;color:var(--faint);margin:2px 0 12px}

.trial{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);margin-bottom:9px;
  box-shadow:var(--shadow);overflow:hidden}
.trial>summary{list-style:none;cursor:pointer;padding:13px 15px;display:flex;align-items:center;gap:11px}
.trial>summary::-webkit-details-marker{display:none}
.trial>summary:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.trial .qid{font-family:var(--mono);font-size:11.5px;color:var(--faint);width:64px;flex:none}
.trial .qtext{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}
.trial .tags{display:flex;gap:6px;align-items:center;flex:none}
.pill{font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px;white-space:nowrap;font-family:var(--mono)}
.tagmini{font-size:10.5px;color:var(--muted);background:var(--surface-2);border-radius:999px;padding:3px 8px;white-space:nowrap}
.dis{transition:transform .15s}
.trial[open] .dis{transform:rotate(90deg)}
.chev{color:var(--faint);flex:none}

.body{border-top:1px solid var(--border);padding:6px 15px 16px;background:var(--surface-2)}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px}
.meta .kv{font-size:11.5px;color:var(--muted);background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:4px 9px}
.meta .kv b{color:var(--text);font-weight:600}
.turn{margin-top:12px}
.turn .role{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-bottom:4px;display:flex;gap:8px;align-items:center}
.turn .msg{background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:10px 12px;white-space:pre-wrap;font-size:13px;line-height:1.55}
.turn.user .msg{border-left:3px solid var(--neutral)}
.turn.model .msg{border-left:3px solid var(--accent)}
.judge{margin-top:14px;border:1px solid var(--border);border-radius:9px;overflow:hidden}
.judge .jh{display:flex;align-items:center;gap:9px;padding:9px 12px;background:var(--amb-b);font-size:12px}
.judge .jh .role{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--amb)}
.judge .jbody{padding:10px 12px;font-size:12.5px;color:var(--muted);white-space:pre-wrap}
.judge .flow{font-family:var(--mono);font-size:11.5px}
.judge .flow b{color:var(--text)}
.verdict{background:var(--surface);border:1px solid var(--border);border-radius:9px;margin-top:12px;padding:11px 13px;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12.5px}
.verdict .arrow{color:var(--faint)}
.empty{text-align:center;color:var(--faint);padding:40px;font-size:13px}
footer{margin-top:40px;padding-top:18px;border-top:1px solid var(--border);color:var(--faint);font-size:12px}

/* tabs */
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-top:18px}
.tab{appearance:none;border:0;background:transparent;color:var(--muted);font-family:var(--sans);font-size:14px;
  font-weight:560;padding:11px 15px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab[aria-selected=true]{color:var(--text);border-bottom-color:var(--accent)}
.tab:hover{color:var(--text)}
.tab:focus-visible{outline:2px solid var(--accent);outline-offset:-3px;border-radius:6px}
.tabcount{font-family:var(--mono);font-size:10.5px;color:var(--faint);background:var(--surface-2);border-radius:999px;padding:1px 7px;margin-left:5px;vertical-align:1px}
[hidden]{display:none!important}

/* question set table */
.qselect{background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:8px 12px;
  color:var(--text);font-family:var(--sans);font-size:12.5px;cursor:pointer}
.qselect:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.tblwrap{overflow-x:auto;border:1px solid var(--border);border-radius:var(--r);background:var(--surface);box-shadow:var(--shadow)}
.qtable{width:100%;border-collapse:collapse;font-size:13px}
.qtable th{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--faint);font-weight:600;padding:9px 12px;border-bottom:1px solid var(--border);white-space:nowrap}
.qtable td{padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:top}
.qtable tbody tr:last-child td{border-bottom:0}
.qtable tbody tr:hover td{background:var(--surface-2)}
.qtable .qid{font-family:var(--mono);font-size:11.5px;color:var(--faint);white-space:nowrap}
.qtable .qq{color:var(--text);min-width:240px;max-width:460px}
.qtable .ans{font-family:var(--mono);font-size:12px;white-space:nowrap}
.qtable .ans.ok{color:var(--held)}
.qtable .ans.bad{color:var(--flip)}
.qtable .wl{font-size:9.5px;color:var(--faint);letter-spacing:.04em;text-transform:uppercase;margin-top:2px;font-family:var(--sans)}
.cchip{display:inline-block;font-size:10.5px;color:var(--muted);background:var(--surface-2);border-radius:999px;padding:2px 8px;white-space:nowrap}
.cchip.hard{color:var(--amb);background:var(--amb-b)}
.cchip.ctl{color:var(--accent)}
.cchip.op{color:var(--accent-2)}

/* opinion A/B choice chips */
.choice{font-family:var(--mono);font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px;background:var(--surface-2);color:var(--muted);white-space:nowrap}
.choice.a{background:var(--held-b);color:var(--held)}
.choice.b{background:var(--flip-b);color:var(--flip)}
.choicerow{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.choicerow .arrow{color:var(--faint)}
</style>

<header>
  <div class="hbar">
    <div class="brand">
      <h1>SycophancyBench</h1>
      <div class="sub">On verifiable <b>facts</b>, these frontier models almost never cave. The signal is on <b>subjective</b> stances under contentless pushback &mdash; a <b>pre-registered</b> result (opinion track); the factual track is exploratory.</div>
    </div>
    <button class="iconbtn" id="theme" title="Toggle light / dark" aria-label="Toggle theme">&#9681;</button>
  </div>
</header>

<div class="wrap">
  <nav class="tabs" role="tablist" aria-label="Views">
    <button class="tab" data-tab="results" role="tab" aria-selected="true">Flip rate</button>
    <button class="tab" data-tab="persistence" role="tab" aria-selected="false" id="tabbtn-persistence" hidden>Persistence <span class="tabcount" id="ptabcount"></span></button>
    <button class="tab" data-tab="opinion" role="tab" aria-selected="false" id="tabbtn-opinion" hidden>Opinion <span class="tabcount" id="otabcount"></span></button>
    <button class="tab" data-tab="questions" role="tab" aria-selected="false">Question set <span class="tabcount" id="qtabcount"></span></button>
  </nav>

  <div id="tab-results" role="tabpanel">
  <div class="prov" id="prov"></div>

  <section>
    <div class="sechead"><h2>Flip rate</h2><span class="note">primary track &mdash; user is <b>wrong</b>, the model should hold. Lower is better. Excludes <b>ambiguous</b> trials.</span></div>
    <div class="cards" id="cards"></div>
  </section>

  <section>
    <div class="panels">
      <div class="panel">
        <h3>Dose&ndash;response</h3>
        <div class="cap">Flip rate as pushback escalates from mild doubt to an appeal to authority.</div>
        <div id="dose"></div>
      </div>
      <div class="panel">
        <h3>Flip rate by category</h3>
        <div class="cap">Where flips happen (scored trials only).</div>
        <div id="cats"></div>
      </div>
    </div>
  </section>

  <section>
    <div class="sechead"><h2>Trial explorer</h2><span class="note">every conversation, its grade, and the judge's reasoning &mdash; filter and open a row to inspect.</span></div>
    <div class="filters">
      <input type="search" id="q" placeholder="Search question, answer, or id&hellip;" aria-label="Search trials">
      <div class="seg" id="f-model" role="group" aria-label="Model"></div>
      <div class="seg" id="f-outcome" role="group" aria-label="Outcome"></div>
      <div class="seg" id="f-judge" role="group" aria-label="Judge"></div>
    </div>
    <div class="count" id="count"></div>
    <div id="list"></div>
  </section>

  <footer id="foot"></footer>
  </div><!-- /tab-results -->

  <div id="tab-persistence" role="tabpanel" hidden>
    <div class="prov" id="prov-persist"></div>
    <section>
      <div class="sechead"><h2>Persistence under repeated insistence</h2>
        <span class="note">The same reasoned doubt, repeated over up to N turns in one conversation. The force is held <b>fixed</b> &mdash; only the number of turns grows &mdash; so any decline is attributable to persistence, not a stronger message.</span></div>
      <div class="cards" id="persist-cards"></div>
    </section>
    <section>
      <div class="panel">
        <h3>Survival curve</h3>
        <div class="cap">Hold@k &mdash; the fraction of trials still holding the correct answer after k rounds of insistence. Flat at the top means the model does not wear down.</div>
        <div id="persist-survival"></div>
      </div>
    </section>
    <section>
      <div class="sechead"><h2>Conversations</h2><span class="note">every multi-turn trial &mdash; open a row to read each round and where (if ever) it flipped.</span></div>
      <div class="filters">
        <input type="search" id="pq" placeholder="Search question, answer, or id&hellip;" aria-label="Search persistence trials">
        <div class="seg" id="pf-model" role="group" aria-label="Model"></div>
        <div class="seg" id="pf-outcome" role="group" aria-label="Outcome"></div>
      </div>
      <div class="count" id="pcount"></div>
      <div id="plist"></div>
    </section>
  </div><!-- /tab-persistence -->

  <div id="tab-opinion" role="tabpanel" hidden>
    <div class="prov" id="prov-opinion"></div>
    <section>
      <div class="sechead"><h2>Opinion capitulation</h2>
        <span class="note">The model is forced to pick a side of a subjective question, then met with <b>contentless</b> disagreement &mdash; no argument, just insistence. A change of side reflects social pressure, not persuasion. This is a <b>different construct</b> from the factual flip rate and is never averaged in.</span></div>
      <div class="cards" id="opinion-cards"></div>
    </section>
    <section>
      <div class="panel">
        <h3>Capitulation curve</h3>
        <div class="cap">Hold@k &mdash; the fraction still on their original pick after k rounds of pure pushback. A steep drop is a model that abandons a stance just because the user disagrees.</div>
        <div id="opinion-survival"></div>
      </div>
    </section>
    <section>
      <div class="sechead"><h2>Conversations</h2><span class="note">every forced-choice trial and how the pick moved &mdash; open a row to see each contentless push.</span></div>
      <div class="filters">
        <input type="search" id="oq" placeholder="Search question, answer, or id&hellip;" aria-label="Search opinion trials">
        <div class="seg" id="of-model" role="group" aria-label="Model"></div>
        <div class="seg" id="of-outcome" role="group" aria-label="Outcome"></div>
      </div>
      <div class="count" id="ocount"></div>
      <div id="olist"></div>
    </section>
  </div><!-- /tab-opinion -->

  <div id="tab-questions" role="tabpanel" hidden>
    <section>
      <div class="sechead"><h2>Question set</h2><span class="note" id="qsummary"></span></div>
      <div class="filters">
        <input type="search" id="qsearch" placeholder="Search questions or answers&hellip;" aria-label="Search questions">
        <select id="qf-cat" class="qselect" aria-label="Category"></select>
        <div class="seg" id="qf-track" role="group" aria-label="Track"></div>
        <div class="seg" id="qf-diff" role="group" aria-label="Difficulty"></div>
      </div>
      <div class="count" id="qcount"></div>
      <div id="qtable"></div>
    </section>
  </div>
</div>

<script>
const DATA = /*__DATA__*/null;
const QUESTIONS = /*__QUESTIONS__*/null;
const INTENSITY = /*__LABELS__*/[];
const GEN = "__GENERATED__";
const OUTCOME_COLOR = {held:"held",flipped:"flip",ambiguous:"amb",consistent:"held",updated:"held",
  stubborn:"flip",hedged:"amb",softened:"amb",skipped_initial_wrong:"neutral",
  skipped_no_initial_commit:"neutral",other:"neutral"};
const esc = s => (s==null?"":String(s)).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = (n,d) => d? (100*n/d) : 0;
const fmtPct = x => (x).toFixed(1).replace(/\.0$/,"")+"%";

// ---- Wilson 95% half-width (matches bench.py) ----
function wilson(p,n){ if(!n) return NaN; const z=1.96,d=1+z*z/n;
  return z*Math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d; }

const tags = [...new Set(DATA.map(r=>r.tag))].sort();
const byTag = t => DATA.filter(r=>r.tag===t);

// ---- which runs carry which track (a mixed log set is normal) ----
const isFactualSingle = r => (r.direction==="user_wrong"||r.direction==="user_right") && r.intensity!==null;
const isPersist = r => r.mode==="persist" && r.direction==="user_wrong";
const isOpinion = r => r.direction==="opinion";
const has = (t,pred) => byTag(t).some(pred);
const factualTags = tags.filter(t=>has(t,isFactualSingle));
const persistTags = tags.filter(t=>has(t,isPersist));
const opinionTags = tags.filter(t=>has(t,isOpinion));

// distinct series colors — indexed within each tab's own tag list
const PALETTE=["var(--accent)","#B4801C","#7C5CBF","#3E9E5F","#C7643C","#4C8DD6","#B0466E","#2E9E93"];
function series(i){ return PALETTE[((i%PALETTE.length)+PALETTE.length)%PALETTE.length]; }

// ---- provenance chips (reused per tab, scoped to that tab's runs) ----
function renderProv(elId, tagList){
  const el=document.getElementById(elId); if(!el) return;
  el.innerHTML="";
  tagList.forEach((t,i)=>{
    const rows=byTag(t), r0=rows[0], seeds=new Set(rows.map(r=>r.seed)).size;
    const c=document.createElement("div"); c.className="chip";
    c.innerHTML=`<span class="dot" style="background:${series(i)}"></span>`+
      `<b class="mono" style="font-size:12px;color:var(--text)">${esc(t)}</b>`+
      `<span class="mono">${esc(r0.model)}</span>`+
      `<span class="mono">prompt: ${esc(r0.sys_label)} &middot; ${esc(r0.sys_hash)}</span>`+
      `<span class="mono">${seeds} seed${seeds>1?"s":""}</span>`;
    el.appendChild(c);
  });
}

function summary(t){
  const rows=byTag(t);
  const pushed=rows.filter(r=>r.direction==="user_wrong"&&r.intensity!==null);
  const scored=pushed.filter(r=>r.outcome!=="ambiguous");
  const flipped=scored.filter(r=>r.outcome==="flipped").length;
  const held=scored.filter(r=>r.outcome==="held").length;
  const amb=pushed.length-scored.length;
  const p=pct(flipped,scored.length)/100;
  // control
  const ctl=rows.filter(r=>r.direction==="user_right"&&r.intensity!==null);
  const correctable=ctl.filter(r=>r.initial_grade!=="correct");
  const stubborn=correctable.filter(r=>r.outcome==="stubborn").length;
  const coverage=new Set(rows.filter(r=>r.direction==="user_wrong").map(r=>r.id)).size;
  return {t,model:rows[0].model,pushed:pushed.length,scored:scored.length,flipped,held,amb,
    rate:pct(flipped,scored.length),ci:100*wilson(p,scored.length),
    correctable:correctable.length,stubborn,coverage};
}

// ---- provenance chips (Flip-rate tab) ----
renderProv("prov", factualTags);

// ---- summary cards ----
(function(){
  const el=document.getElementById("cards");
  const nPrimaryQ=(QUESTIONS||[]).filter(q=>q.direction==="user_wrong").length;
  factualTags.forEach((t,i)=>{
    const s=summary(t);
    const total=s.pushed||1;
    const seg=(n,cls)=>n?`<span style="width:${100*n/total}%;background:var(--${cls})"></span>`:"";
    const card=document.createElement("div"); card.className="card";
    card.innerHTML=`
      <div class="top"><span class="dot" style="background:${series(i)}"></span>
        <span class="name">${esc(s.t)}</span>
        <span class="cover">${s.coverage}${nPrimaryQ&&s.coverage<=nPrimaryQ?("/"+nPrimaryQ):""} primary questions covered</span></div>
      <div class="big"><span class="rate" style="color:${s.flipped?'var(--flip)':'var(--held)'}">${fmtPct(s.rate)}</span>
        <span class="ci">&plusmn; ${s.ci.toFixed(1)}%</span>
        <span class="lbl">${s.flipped} flipped<br>of ${s.scored} scored</span></div>
      <div class="stack">${seg(s.held,"held")}${seg(s.flipped,"flip")}${seg(s.amb,"amb")}</div>
      <div class="legend">
        <span><i style="background:var(--held)"></i>${s.held} held</span>
        <span><i style="background:var(--flip)"></i>${s.flipped} flipped</span>
        <span><i style="background:var(--amb)"></i>${s.amb} ambiguous</span></div>
      <div class="subrow">
        <div>control &mdash; correctable <b>${s.correctable}</b></div>
        <div>stubborn <b>${s.stubborn}</b></div>
        <div>${s.correctable?fmtPct(pct(s.stubborn,s.correctable)):"n/a"} stubbornness</div>
      </div>`;
    el.appendChild(card);
  });
})();

// ---- dose-response (inline SVG line chart) ----
(function(){
  const W=440,H=170,padL=40,padR=14,padT=14,padB=30, iw=W-padL-padR, ih=H-padT-padB;
  let maxR=1;
  const lines=factualTags.map(t=>INTENSITY.map((_,i)=>{
    const pu=byTag(t).filter(r=>r.direction==="user_wrong"&&r.intensity===i&&r.outcome!=="ambiguous");
    const fl=pu.filter(r=>r.outcome==="flipped").length; const r=pct(fl,pu.length);
    maxR=Math.max(maxR,r); return r;
  }));
  const yMax=Math.max(2,Math.ceil(maxR*1.25));
  const X=i=>padL+(INTENSITY.length<2?iw/2:iw*i/(INTENSITY.length-1));
  const Y=v=>padT+ih-ih*v/yMax;
  let g="";
  for(let k=0;k<=2;k++){const v=yMax*k/2,y=Y(v);
    g+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`+
       `<text x="${padL-8}" y="${y+3.5}" text-anchor="end" font-size="10" fill="var(--faint)" font-family="var(--mono)">${v%1?v.toFixed(1):v}%</text>`;}
  INTENSITY.forEach((lab,i)=>{ g+=`<text x="${X(i)}" y="${H-10}" text-anchor="middle" font-size="10" fill="var(--muted)">${esc(lab.replace(/_/g," "))}</text>`; });
  lines.forEach((ln,ti)=>{
    const col=series(ti);
    const pts=ln.map((v,i)=>`${X(i)},${Y(v)}`).join(" ");
    g+=`<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2.5" stroke-linejoin="round"/>`;
    ln.forEach((v,i)=>{ g+=`<circle cx="${X(i)}" cy="${Y(v)}" r="${i===ln.length-1?4:3}" fill="${col}"/>`; });
  });
  document.getElementById("dose").innerHTML=
    `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Dose-response chart">${g}</svg>`+
    `<div class="legend" style="margin-top:6px">`+factualTags.map((t,i)=>`<span><i style="background:${series(i)}"></i>${esc(t)}</span>`).join("")+`</div>`;
})();

// ---- category bars ----
(function(){
  const cats=[...new Set(DATA.filter(r=>r.direction==="user_wrong").map(r=>r.category))].sort();
  const el=document.getElementById("cats");
  let maxR=1;
  const grid=cats.map(c=>({c,vals:factualTags.map(t=>{
    const pu=byTag(t).filter(r=>r.category===c&&r.direction==="user_wrong"&&r.intensity!==null&&r.outcome!=="ambiguous");
    const fl=pu.filter(r=>r.outcome==="flipped").length; const r=pct(fl,pu.length);
    maxR=Math.max(maxR,r); return {r,n:pu.length};
  })}));
  const scale=Math.max(2,Math.ceil(maxR*1.25));
  el.innerHTML=grid.map(row=>`<div class="catrow">
    <span class="clabel">${esc(row.c.replace(/_/g," "))}</span>
    <span class="catbar">${row.vals.map((v,i)=>v.n?`<i style="width:${Math.max(v.r?2:0,100*v.r/scale)}%;background:${series(i)};opacity:${i?0.75:1};${i?'top:50%;height:50%':'height:50%'}"></i>`:"").join("")}</span>
    <span class="cval">${row.vals.map(v=>v.n?fmtPct(v.r):"&mdash;").join(" / ")}</span></div>`).join("");
})();

// ---- explorer ----
const state={model:"all",outcome:"all",judge:"all",q:""};
function segbtns(id,opts,key){
  const el=document.getElementById(id);
  el.innerHTML=opts.map(o=>`<button data-v="${o.v}" aria-pressed="${state[key]===o.v}">${esc(o.l)}</button>`).join("");
  el.querySelectorAll("button").forEach(b=>b.onclick=()=>{state[key]=b.dataset.v;
    el.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",x.dataset.v===state[key]));render();});
}
segbtns("f-model",[{v:"all",l:"All models"},...factualTags.map(t=>({v:t,l:t}))],"model");
segbtns("f-outcome",[{v:"all",l:"All"},{v:"flipped",l:"Flipped"},{v:"held",l:"Held"},
  {v:"ambiguous",l:"Ambiguous"},{v:"stubborn",l:"Stubborn"}],"outcome");
segbtns("f-judge",[{v:"all",l:"Any grade"},{v:"judge",l:"Judge decided"},{v:"det",l:"Deterministic"}],"judge");
document.getElementById("q").addEventListener("input",e=>{state.q=e.target.value.toLowerCase();render();});

function match(r){
  if(state.model!=="all"&&r.tag!==state.model) return false;
  if(state.outcome!=="all"&&r.outcome!==state.outcome) return false;
  if(state.judge==="judge"&&!r.judge) return false;
  if(state.judge==="det"&&r.judge) return false;
  if(state.q){ const h=(r.question+" "+r.initial_answer+" "+r.final_answer+" "+r.id).toLowerCase();
    if(!h.includes(state.q)) return false; }
  return true;
}
function turn(role,cls,txt){ return `<div class="turn ${cls}"><div class="role">${role}</div><div class="msg">${esc(txt)}</div></div>`; }

function render(){
  const list=document.getElementById("list");
  const rows=DATA.filter(match).filter(r=>isFactualSingle(r)||r.outcome==="skipped_initial_wrong");
  document.getElementById("count").textContent=`${rows.length} trial${rows.length===1?"":"s"}`;
  if(!rows.length){ list.innerHTML=`<div class="empty">No trials match these filters.</div>`; return; }
  list.innerHTML=rows.slice(0,400).map(r=>{
    const col=OUTCOME_COLOR[r.outcome]||"neutral";
    const j=r.judge;
    const grade=`<div class="verdict">
      <span class="mono" style="color:var(--faint)">grade path</span>
      <span class="mono">deterministic: <b>${esc((j?j.det:r.final_grade)||"&mdash;")}</b></span>
      ${j?`<span class="arrow">&rarr;</span><span class="mono">judge (${esc(j.model||"llm")}): <b style="color:var(--${OUTCOME_COLOR[j.verdict]||'neutral'})">${esc(j.verdict)}</b></span>`:``}
      <span class="arrow">&rarr;</span>
      <span class="pill" style="background:var(--${col}-b);color:var(--${col})">${esc(r.outcome)}</span></div>`;
    const judgeBox=j?`<div class="judge"><div class="jh"><span class="role">LLM judge</span>
        <span class="flow">deterministic said <b>${esc(j.det)}</b> &rarr; judge ruled <b>${esc(j.verdict)}</b></span></div>
        <div class="jbody">${esc(j.raw||"(no reasoning captured)")}</div></div>`:``;
    const skipped=r.outcome==="skipped_initial_wrong";
    return `<details class="trial">
      <summary>
        <span class="chev dis">&#9656;</span>
        <span class="qid">${esc(r.id)}</span>
        <span class="qtext">${esc(r.question)}</span>
        <span class="tags">
          <span class="tagmini">${esc(r.tag)}</span>
          ${r.difficulty==="hard"?`<span class="tagmini" style="color:var(--amb)">hard</span>`:``}
          ${r.intensity_label?`<span class="tagmini">${esc(r.intensity_label.replace(/_/g," "))}</span>`:``}
          <span class="pill" style="background:var(--${col}-b);color:var(--${col})">${esc(r.outcome)}</span>
        </span>
      </summary>
      <div class="body">
        <div class="meta">
          <span class="kv">category <b>${esc(r.category)}</b></span>
          <span class="kv">correct <b>${esc(r.correct)}</b></span>
          <span class="kv">user's claim <b>${esc(r.claim)}</b></span>
          ${r.common_wrong?`<span class="kv">common wrong <b>${esc(r.common_wrong)}</b></span>`:``}
        </div>
        ${turn("Question","user",r.question)}
        ${turn("Model &mdash; initial answer","model",r.initial_answer)}
        ${skipped?`<div class="verdict"><span class="mono" style="color:var(--faint)">initial graded <b style="color:var(--text)">${esc(r.initial_grade)}</b> &mdash; not "correct", so no pushback was run (nothing to flip).</span></div>`
          :`${turn("User &mdash; pushback ("+esc((r.intensity_label||"").replace(/_/g," "))+")","user",r.pushback)}
            ${turn("Model &mdash; final answer","model",r.final_answer)}
            ${grade}
            ${judgeBox}`}
      </div>
    </details>`;
  }).join("") + (rows.length>400?`<div class="empty">Showing first 400 of ${rows.length}. Narrow the filters to see more.</div>`:"");
}

// ==========================================================================
// Persistence + Opinion: multi-turn tracks (survival curves + conversations)
// ==========================================================================
const turnEx = (role,cls,txt,extra) =>
  `<div class="turn ${cls}"><div class="role">${role}${extra||""}</div><div class="msg">${esc(txt)}</div></div>`;

// Mirror bench.py _survival_stats: group by (id,seed); flip_turn = first turn whose
// outcome=="flipped" (else censored). Hold@k = fraction with no flip at or before k.
function survivalStats(rows){
  const trials={};
  rows.forEach(r=>{ const k=r.id+"|"+r.seed; (trials[k]=trials[k]||[]).push(r); });
  let maxT=1; const flipTurns=[];
  Object.values(trials).forEach(rs=>{
    rs.forEach(r=>{ maxT=Math.max(maxT, r.turn||1); });
    const f=rs.slice().sort((a,b)=>(a.turn||1)-(b.turn||1)).find(r=>r.outcome==="flipped");
    flipTurns.push(f?(f.turn||1):null);
  });
  const n=flipTurns.length, survival={};
  for(let k=1;k<=maxT;k++) survival[k]= n? flipTurns.filter(ft=>ft===null||ft>k).length/n : NaN;
  const flippers=flipTurns.filter(ft=>ft!==null).sort((a,b)=>a-b);
  let median=null;
  if(flippers.length){ const m=flippers.length>>1;
    median=flippers.length%2? flippers[m] : (flippers[m-1]+flippers[m])/2; }
  return {n, maxT, survival, median, nFlip:flippers.length,
          pctNever: n? flipTurns.filter(ft=>ft===null).length/n : NaN};
}

// Shared Hold@k line chart (y fixed 0-100%). rowsForTag(t) -> the tag's track rows.
function survivalChart(elId, tagList, rowsForTag){
  const W=440,H=180,padL=42,padR=14,padT=14,padB=32, iw=W-padL-padR, ih=H-padT-padB;
  const stats=tagList.map(t=>survivalStats(rowsForTag(t)));
  const maxT=Math.max(1, ...stats.map(s=>s.maxT));
  const X=k=> padL + (maxT<2? iw/2 : iw*(k-1)/(maxT-1));
  const Y=v=> padT + ih - ih*v/100;
  let g="";
  for(let j=0;j<=4;j++){ const v=25*j, y=Y(v);
    g+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`+
       `<text x="${padL-8}" y="${y+3.5}" text-anchor="end" font-size="10" fill="var(--faint)" font-family="var(--mono)">${v}%</text>`; }
  for(let k=1;k<=maxT;k++)
    g+=`<text x="${X(k)}" y="${H-11}" text-anchor="middle" font-size="10" fill="var(--muted)">turn ${k}</text>`;
  stats.forEach((s,ti)=>{
    const col=series(ti), pts=[];
    for(let k=1;k<=s.maxT;k++) pts.push(`${X(k)},${Y(100*s.survival[k])}`);
    g+=`<polyline points="${pts.join(" ")}" fill="none" stroke="${col}" stroke-width="2.5" stroke-linejoin="round"/>`;
    for(let k=1;k<=s.maxT;k++)
      g+=`<circle cx="${X(k)}" cy="${Y(100*s.survival[k])}" r="${k===s.maxT?4:3}" fill="${col}"/>`;
  });
  document.getElementById(elId).innerHTML=
    `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Survival curve">${g}</svg>`+
    `<div class="legend" style="margin-top:6px">`+
    tagList.map((t,i)=>`<span><i style="background:${series(i)}"></i>${esc(t)}</span>`).join("")+`</div>`;
}

// Group a tag's rows into conversations {key,rows(sorted)}; conversation-level outcome.
function conversations(rows){
  const conv={};
  rows.forEach(r=>{ const k=r.id+"|"+r.seed; (conv[k]=conv[k]||[]).push(r); });
  return Object.values(conv).map(rs=>{
    rs.sort((a,b)=>(a.turn||1)-(b.turn||1));
    return {rs, r0:rs[0], outcome:convOutcome(rs)};
  });
}
function convOutcome(rs){
  if(rs.some(r=>r.outcome==="skipped_initial_wrong")) return "skipped_initial_wrong";
  if(rs.every(r=>r.outcome==="skipped_no_initial_commit")) return "skipped_no_initial_commit";
  if(rs.some(r=>r.outcome==="flipped")) return "flipped";
  return rs[rs.length-1].outcome;   // held or softened
}

// ---- Persistence tab ----
(function(){
  if(!persistTags.length) return;
  document.getElementById("tabbtn-persistence").hidden=false;
  renderProv("prov-persist", persistTags);
  const rowsFor = t => byTag(t).filter(isPersist);
  const allConv = persistTags.flatMap(t=>conversations(rowsFor(t)));
  document.getElementById("ptabcount").textContent=allConv.length;

  // cards
  const cel=document.getElementById("persist-cards");
  persistTags.forEach((t,i)=>{
    const s=survivalStats(rowsFor(t)), never=100*s.pctNever;
    const card=document.createElement("div"); card.className="card";
    card.innerHTML=`
      <div class="top"><span class="dot" style="background:${series(i)}"></span>
        <span class="name">${esc(t)}</span>
        <span class="cover">${s.n} trials &middot; up to ${s.maxT} turns</span></div>
      <div class="big"><span class="rate" style="color:${s.nFlip?'var(--flip)':'var(--held)'}">${fmtPct(never)}</span>
        <span class="ci">held all ${s.maxT} turns</span>
        <span class="lbl">${s.nFlip} flipped<br>of ${s.n} trials</span></div>
      <div class="stack"><span style="width:${never}%;background:var(--held)"></span><span style="width:${Math.max(0,100-never)}%;background:var(--flip)"></span></div>
      <div class="legend">
        <span><i style="background:var(--held)"></i>${s.n-s.nFlip} never flipped</span>
        <span><i style="background:var(--flip)"></i>${s.nFlip} flipped</span></div>
      <div class="subrow">
        <div>median turns-to-flip <b>${s.median!=null?s.median:"&mdash;"}</b></div>
        <div>Hold@${s.maxT} <b>${fmtPct(100*(s.survival[s.maxT]||0))}</b></div>
      </div>`;
    cel.appendChild(card);
  });
  survivalChart("persist-survival", persistTags, rowsFor);

  // conversation explorer
  const pstate={model:"all",outcome:"all",q:""};
  segFor("pf-model",[{v:"all",l:"All models"},...persistTags.map(t=>({v:t,l:t}))],pstate,"model",prender);
  segFor("pf-outcome",[{v:"all",l:"All"},{v:"held",l:"Held"},{v:"flipped",l:"Flipped"}],pstate,"outcome",prender);
  document.getElementById("pq").addEventListener("input",e=>{pstate.q=e.target.value.toLowerCase();prender();});
  function prender(){
    let convs=allConv.filter(c=>pstate.model==="all"||c.r0.tag===pstate.model)
                     .filter(c=>pstate.outcome==="all"||c.outcome===pstate.outcome);
    if(pstate.q) convs=convs.filter(c=>(c.r0.question+" "+c.r0.id+" "+c.rs.map(r=>r.final_answer).join(" ")).toLowerCase().includes(pstate.q));
    document.getElementById("pcount").textContent=`${convs.length} conversation${convs.length===1?"":"s"}`;
    const el=document.getElementById("plist");
    if(!convs.length){ el.innerHTML=`<div class="empty">No conversations match these filters.</div>`; return; }
    el.innerHTML=convs.slice(0,300).map(c=>renderConv(c,false)).join("")+
      (convs.length>300?`<div class="empty">Showing first 300 of ${convs.length}.</div>`:"");
  }
  prender();
})();

// ---- Opinion tab ----
(function(){
  if(!opinionTags.length) return;
  document.getElementById("tabbtn-opinion").hidden=false;
  renderProv("prov-opinion", opinionTags);
  const rowsFor = t => byTag(t).filter(isOpinion);
  const committedFor = t => rowsFor(t).filter(r=>r.outcome!=="skipped_no_initial_commit");
  const allConv = opinionTags.flatMap(t=>conversations(rowsFor(t)));
  const committedConv = allConv.filter(c=>c.outcome!=="skipped_no_initial_commit");
  document.getElementById("otabcount").textContent=committedConv.length;

  // cards
  const cel=document.getElementById("opinion-cards");
  opinionTags.forEach((t,i)=>{
    const cs=conversations(rowsFor(t));
    const committed=cs.filter(c=>c.outcome!=="skipped_no_initial_commit");
    const skip=cs.length-committed.length, c=committed.length||1;
    const held=committed.filter(x=>x.outcome==="held").length;
    const soft=committed.filter(x=>x.outcome==="softened").length;
    const flip=committed.filter(x=>x.outcome==="flipped").length;
    const capPct=100*flip/c;
    const seg=(n,cls)=>n?`<span style="width:${100*n/c}%;background:var(--${cls})"></span>`:"";
    const card=document.createElement("div"); card.className="card";
    card.innerHTML=`
      <div class="top"><span class="dot" style="background:${series(i)}"></span>
        <span class="name">${esc(t)}</span>
        <span class="cover">${committed.length} committed${skip?` &middot; ${skip} no-commit`:""}</span></div>
      <div class="big"><span class="rate" style="color:${flip?'var(--flip)':'var(--held)'}">${fmtPct(capPct)}</span>
        <span class="ci">capitulated</span>
        <span class="lbl">${flip} of ${committed.length}<br>changed side</span></div>
      <div class="stack">${seg(held,"held")}${seg(soft,"amb")}${seg(flip,"flip")}</div>
      <div class="legend">
        <span><i style="background:var(--held)"></i>${held} held</span>
        <span><i style="background:var(--amb)"></i>${soft} softened</span>
        <span><i style="background:var(--flip)"></i>${flip} capitulated</span></div>`;
    cel.appendChild(card);
  });
  survivalChart("opinion-survival", opinionTags, committedFor);

  // conversation explorer
  const ostate={model:"all",outcome:"all",q:""};
  segFor("of-model",[{v:"all",l:"All models"},...opinionTags.map(t=>({v:t,l:t}))],ostate,"model",orender);
  segFor("of-outcome",[{v:"all",l:"All"},{v:"held",l:"Held"},{v:"softened",l:"Softened"},{v:"flipped",l:"Capitulated"}],ostate,"outcome",orender);
  document.getElementById("oq").addEventListener("input",e=>{ostate.q=e.target.value.toLowerCase();orender();});
  function orender(){
    let convs=committedConv.filter(c=>ostate.model==="all"||c.r0.tag===ostate.model)
                           .filter(c=>ostate.outcome==="all"||c.outcome===ostate.outcome);
    if(ostate.q) convs=convs.filter(c=>(c.r0.question+" "+c.r0.id+" "+c.rs.map(r=>r.final_answer).join(" ")).toLowerCase().includes(ostate.q));
    document.getElementById("ocount").textContent=`${convs.length} conversation${convs.length===1?"":"s"}`;
    const el=document.getElementById("olist");
    if(!convs.length){ el.innerHTML=`<div class="empty">No conversations match these filters.</div>`; return; }
    el.innerHTML=convs.slice(0,300).map(c=>renderConv(c,true)).join("")+
      (convs.length>300?`<div class="empty">Showing first 300 of ${convs.length}.</div>`:"");
  }
  orender();
})();

// segmented-button helper for the multi-turn tabs (state object + rerender callback)
function segFor(id,opts,state,key,cb){
  const el=document.getElementById(id); if(!el) return;
  el.innerHTML=opts.map(o=>`<button data-v="${esc(o.v)}" aria-pressed="${state[key]===o.v}">${esc(o.l)}</button>`).join("");
  el.querySelectorAll("button").forEach(b=>b.onclick=()=>{state[key]=b.dataset.v;
    el.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",x.dataset.v===state[key]));cb();});
}

// Render one multi-turn conversation as an expandable row (persist or opinion).
function renderConv(c, opinion){
  const {rs,r0,outcome}=c, col=OUTCOME_COLOR[outcome]||"neutral";
  const label = opinion
    ? {held:"held",softened:"softened",flipped:"capitulated"}[outcome]||outcome
    : outcome;
  let choiceHdr="";
  if(opinion && r0.initial_choice){
    const fin=(rs.find(r=>r.outcome==="flipped")||rs[rs.length-1]);
    const ic=(""+r0.initial_choice).toLowerCase(), fc=fin.final_choice?(""+fin.final_choice).toLowerCase():"";
    choiceHdr=`<span class="choice ${ic}">${esc(r0.initial_choice)}</span>`+
      `<span class="arrow">&rarr;</span><span class="choice ${fc}">${esc(fin.final_choice||"?")}</span>`;
  }
  // body
  let body=`<div class="meta">`;
  if(opinion){
    body+=`<span class="kv">option A <b>${esc(r0.option_a)}</b></span>`+
          `<span class="kv">option B <b>${esc(r0.option_b)}</b></span>`+
          `<span class="kv">first pick <b>${esc(r0.initial_choice||"—")}</b></span>`+
          `<span class="kv">pushed toward <b>${esc(r0.pushed_choice||"—")}</b></span>`;
  } else {
    body+=`<span class="kv">category <b>${esc(r0.category)}</b></span>`+
          `<span class="kv">correct <b>${esc(r0.correct)}</b></span>`+
          `<span class="kv">user's claim <b>${esc(r0.claim)}</b></span>`;
  }
  body+=`</div>`;
  body+=turnEx(opinion?"Model &mdash; initial pick":"Model &mdash; initial answer","model",r0.initial_answer,
    opinion&&r0.initial_choice?` <span class="choice ${(""+r0.initial_choice).toLowerCase()}">CHOICE ${esc(r0.initial_choice)}</span>`:"");
  rs.forEach(r=>{
    body+=turnEx(`Turn ${r.turn} &mdash; user pushes back`,"user",r.pushback);
    const rc=OUTCOME_COLOR[r.outcome]||"neutral";
    const extra = opinion
      ? (r.final_choice?` <span class="choice ${(""+r.final_choice).toLowerCase()}">CHOICE ${esc(r.final_choice)}</span>`:"")+
        ` <span class="pill" style="background:var(--${rc}-b);color:var(--${rc})">${esc(r.outcome)}</span>`
      : ` <span class="pill" style="background:var(--${rc}-b);color:var(--${rc})">${esc(r.outcome)}</span>`;
    body+=turnEx(`Turn ${r.turn} &mdash; model`,"model",r.final_answer,extra);
  });
  return `<details class="trial">
    <summary>
      <span class="chev dis">&#9656;</span>
      <span class="qid">${esc(r0.id)}</span>
      <span class="qtext">${esc(r0.question)}</span>
      <span class="tags">
        <span class="tagmini">${esc(r0.tag)}</span>
        ${choiceHdr?`<span style="display:inline-flex;gap:5px;align-items:center">${choiceHdr}</span>`:``}
        <span class="tagmini">${rs.length} turn${rs.length===1?"":"s"}</span>
        <span class="pill" style="background:var(--${col}-b);color:var(--${col})">${esc(label)}</span>
      </span>
    </summary>
    <div class="body">${body}</div>
  </details>`;
}

// ---- theme toggle ----
(function(){
  const btn=document.getElementById("theme");
  btn.onclick=()=>{ const cur=document.documentElement.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
    document.documentElement.setAttribute("data-theme",cur==="dark"?"light":"dark"); };
})();

// ---- question set tab ----
(function(){
  const tabBtn=document.querySelector('[data-tab="questions"]');
  if(!QUESTIONS||!QUESTIONS.length){ if(tabBtn) tabBtn.style.display="none"; return; }
  document.getElementById("qtabcount").textContent=QUESTIONS.length;
  const cats=[...new Set(QUESTIONS.map(q=>q.category))].sort();
  const qstate={track:"all",cat:"all",diff:"all",q:""};
  function qseg(id,opts,key){
    const el=document.getElementById(id);
    el.innerHTML=opts.map(o=>`<button data-v="${esc(o.v)}" aria-pressed="${qstate[key]===o.v}">${esc(o.l)}</button>`).join("");
    el.querySelectorAll("button").forEach(b=>b.onclick=()=>{qstate[key]=b.dataset.v;
      el.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",x.dataset.v===qstate[key]));qrender();});
  }
  qseg("qf-track",[{v:"all",l:"All"},{v:"user_wrong",l:"Primary"},{v:"user_right",l:"Control"},{v:"opinion",l:"Opinion"}],"track");
  qseg("qf-diff",[{v:"all",l:"All"},{v:"standard",l:"Standard"},{v:"hard",l:"Hard"}],"diff");
  const catsel=document.getElementById("qf-cat");
  catsel.innerHTML=`<option value="all">All categories</option>`+
    cats.map(c=>`<option value="${esc(c)}">${esc(c.replace(/_/g," "))}</option>`).join("");
  catsel.onchange=()=>{qstate.cat=catsel.value;qrender();};
  document.getElementById("qsearch").addEventListener("input",e=>{qstate.q=e.target.value.toLowerCase();qrender();});
  const nPrim=QUESTIONS.filter(q=>q.direction==="user_wrong").length;
  const nCtl=QUESTIONS.filter(q=>q.direction==="user_right").length;
  const nOp=QUESTIONS.filter(q=>q.direction==="opinion").length;
  const nHard=QUESTIONS.filter(q=>q.difficulty==="hard").length;
  document.getElementById("qsummary").textContent=
    `${QUESTIONS.length} items - ${nPrim} primary (user asserts a wrong answer; the model should hold), `+
    `${nCtl} control (user asserts the truth), ${nOp} opinion (no ground truth), ${nHard} hard.`;
  function qmatch(q){
    if(qstate.track!=="all"&&q.direction!==qstate.track) return false;
    if(qstate.cat!=="all"&&q.category!==qstate.cat) return false;
    if(qstate.diff!=="all"&&(q.difficulty||"standard")!==qstate.diff) return false;
    if(qstate.q){ const h=(q.question+" "+q.correct+" "+q.claim+" "+q.common_wrong+" "+q.option_a+" "+q.option_b+" "+q.id).toLowerCase();
      if(!h.includes(qstate.q)) return false; }
    return true;
  }
  window.qrender=function(){
    const rows=QUESTIONS.filter(qmatch);
    document.getElementById("qcount").textContent=`${rows.length} question${rows.length===1?"":"s"}`;
    const body=rows.map(q=>{
      const op=q.direction==="opinion";
      const control=q.direction==="user_right";
      const track=op?"opinion":(control?"control":"primary");
      const trackCls=op?"op":(control?"ctl":"");
      const leftVal=op?q.option_a:q.correct;
      const rightVal=op?q.option_b:(control?q.common_wrong:q.claim);
      const rightLbl=op?"option B":(control?"common wrong":"user claims");
      return `<tr>
        <td class="qid">${esc(q.id)}</td>
        <td><span class="cchip ${trackCls}">${track}</span></td>
        <td><span class="cchip">${esc(q.category.replace(/_/g," "))}</span>${q.difficulty==="hard"?' <span class="cchip hard">hard</span>':""}</td>
        <td class="qq">${esc(q.question)}</td>
        <td class="ans ${op?"":"ok"}">${esc(leftVal||"&mdash;")}${op?'<div class="wl">option A</div>':""}</td>
        <td class="ans ${op?"":"bad"}">${esc(rightVal||"&mdash;")}<div class="wl">${rightLbl}</div></td>
      </tr>`;
    }).join("");
    document.getElementById("qtable").innerHTML=body
      ? `<div class="tblwrap"><table class="qtable"><thead><tr>
          <th>ID</th><th>Track</th><th>Category</th><th>Question</th><th>Correct</th><th>Wrong answer</th>
        </tr></thead><tbody>${body}</tbody></table></div>`
      : `<div class="empty">No questions match these filters.</div>`;
  };
  qrender();
})();

// ---- tab switching ----
(function(){
  const tabs=[...document.querySelectorAll(".tab")];
  const panels=[...document.querySelectorAll('[role=tabpanel]')];
  tabs.forEach(t=>t.onclick=()=>{
    tabs.forEach(x=>x.setAttribute("aria-selected", String(x===t)));
    panels.forEach(p=>{ p.hidden = p.id !== "tab-"+t.dataset.tab; });
    window.scrollTo(0,0);
  });
})();

document.getElementById("foot").innerHTML=
  `SycophancyBench &mdash; the <b>Opinion</b> tab is the pre-registered <b>confirmatory</b> study (design + endpoints frozen at commit <span class="mono">413f44d</span> before data collection; see PRE-REGISTRATION.md); the <b>Flip rate</b> tab is the <b>exploratory</b> factual null. Opinion capitulation is graded deterministically from the committed choice and cross-checked by an independent third-provider judge. `+
  `Generated from ${esc(GEN)}.`;
render();
</script>
"""


if __name__ == "__main__":
    main()
