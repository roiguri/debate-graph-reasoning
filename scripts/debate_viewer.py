"""Debate trace viewer: a tiny local server that loads transcripts on demand.

A reading room for "is the Critic useful?": browse Proposer-Critic transcripts against the
actual graph. Serves a lightweight index (id, cell, correct, #turns) and fetches each full
transcript + its graph on demand, so the browser never holds all (~1800) debates at once.
The graph is drawn with Cytoscape (force-directed, seeded positions from the dataset), the
query node lit. Stdlib server (+ networkx, already a dep); localhost.

    python scripts/debate_viewer.py results/debate-pilot        # -> http://localhost:8000
    python scripts/debate_viewer.py results/main --port 8080

Data is snapshotted at startup; restart to pick up new debates from a running job.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from gedebate.data.store import load_dataset
from gedebate.eval import results
from gedebate.graphqa.graph_text_encoder import TEXT_ENCODER_DICT

_ROW_KEYS = ("task", "encoding", "correct", "parse_ok", "parsed_answer",
             "ground_truth", "n_responses", "n_prompt_tokens", "n_gen_tokens")


def load(run_dir: str):
    """Return (rows_by_id, turns_by_id, instances_by_id, lightweight_index)."""
    rows = {r["instance_id"]: r for f in results.result_files(run_dir)
            for r in results.read_rows(f) if r["condition"] == "debate"}
    turns = {t["instance_id"]: t["turns"]
             for f in Path(run_dir).glob("**/*.trace.jsonl") for t in results.read_traces(f)}
    manifest = results.read_manifest(run_dir) or {}
    ds = manifest.get("dataset")
    instances = {}
    if ds and Path(ds).exists():
        instances = {i.instance_id: i for i in load_dataset(ds)}
    index = [{"instance_id": iid, "task": r["task"], "encoding": r["encoding"],
              "correct": bool(r["correct"]), "n_responses": r.get("n_responses", 0)}
             for iid, r in sorted(rows.items())]
    return rows, turns, instances, index


def _component_layout(g):
    """Force-directed positions that stay compact under many disconnected components.

    Plain spring_layout flings each connected component toward a far corner, so
    fitting the whole graph zooms out and crushes every cluster (edges vanish).
    Instead lay each component out on its own, normalize it to a unit box, and tile
    the components into a near-square grid -- every group gets equal, compact area.
    Deterministic (seed=42); identical to before for a single-component graph.
    """
    import math
    import networkx as nx
    comps = sorted(nx.connected_components(g), key=lambda c: (-len(c), min(c)))
    if len(comps) <= 1:
        return nx.spring_layout(g, seed=42)
    ncols = math.ceil(math.sqrt(len(comps)))
    pos = {}
    for idx, comp in enumerate(comps):
        sp = nx.spring_layout(g.subgraph(comp), seed=42)
        xs = [p[0] for p in sp.values()]
        ys = [p[1] for p in sp.values()]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        col, row = idx % ncols, idx // ncols
        for n, (x, y) in sp.items():
            fx = 0.5 if maxx == minx else (x - minx) / (maxx - minx)
            fy = 0.5 if maxy == miny else (y - miny) / (maxy - miny)
            pos[n] = (col + 0.1 + 0.8 * fx, row + 0.1 + 0.8 * fy)  # 0.1 margin inside its cell
    return pos


def _node_labels(inst) -> dict[str, str]:
    """Node id -> the name the model actually read, per encoding.

    The encodings differ mainly in how they *name* nodes (integers for adjacency /
    incident, people for friendship, ...), so the drawing must use the same words as the
    transcript -- otherwise the graph says "1" while the debate argues about "Robert".
    `random` regenerates its names per process (`random.randint` at dict creation), so a
    stored run's names cannot be reproduced here: fall back to the integer id rather than
    print names the run never used. Same fallback for any id past the dictionary's end.
    """
    names = {} if inst.encoding == "random" else TEXT_ENCODER_DICT.get(inst.encoding, {})
    return {str(n): str(names.get(n, n)) for n in range(inst.nnodes)}


def _graph_payload(inst) -> dict | None:
    """Nodes/edges + seeded force-directed positions for Cytoscape (stable per graph)."""
    if inst is None:
        return None
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(inst.nnodes))
    g.add_edges_from(tuple(e) for e in inst.graph_edgelist)
    pos = _component_layout(g)  # component-aware: keeps disconnected clusters compact
    labels = _node_labels(inst)
    # `named` encodings (friendship, ...) draw wide pills; numbered ones (adjacency,
    # incident) keep the original discs and the original 130 spacing, untouched.
    #
    # For names, the layout is normalized, so the scale sets node spacing in the same units
    # as node size. A name ("Christopher") is wide but no taller than an integer, so only
    # the X axis needs the extra room; scaling both would inflate the drawing and, since
    # cy.fit() zooms it into the panel, shrink the text right back.
    named = any(k != v for k, v in labels.items())
    label_w = 7.5 * max(len(v) for v in labels.values()) + 18   # ~13px monospace + padding
    sx = max(130.0, 3.2 * label_w) if named else 130.0
    return {
        "nodes": list(range(inst.nnodes)),
        "edges": [list(e) for e in inst.graph_edgelist],
        "positions": {str(n): {"x": round(float(x) * sx, 1), "y": round(float(y) * 130.0, 1)}
                      for n, (x, y) in pos.items()},
        # every queried node, not just the first: edge_existence asks about a pair
        "query_nodes": list(inst.node_ids),
        "labels": labels,
        "named": named,
        "nnodes": inst.nnodes,
        "encoding_text": inst.question,
    }


def _make_handler(rows, turns, instances, index):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(_PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/api/index":
                return self._send(json.dumps(index).encode(), "application/json")
            if u.path == "/api/trace":
                iid = parse_qs(u.query).get("id", [""])[0]
                r = rows.get(iid)
                if not r:
                    return self.send_error(404)
                payload = {"instance_id": iid, **{k: r[k] for k in _ROW_KEYS},
                           "turns": turns.get(iid, []), "graph": _graph_payload(instances.get(iid))}
                return self._send(json.dumps(payload).encode(), "application/json")
            self.send_error(404)

        def log_message(self, *args):  # quiet
            pass

    return Handler


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Debate Reader</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'><circle cx='20' cy='20' r='20' fill='%230d7268'/><g fill='none' stroke='%23fffdf9' stroke-width='2.1' stroke-linecap='round' stroke-linejoin='round'><circle cx='20' cy='17' r='7'/><path d='M16.6 26.5h6.8M17.6 29.5h4.8'/><path d='M20 13.6v3.4M17.9 18.6h4.2'/></g></svg>">
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<style>
  :root{
    --paper:#f4f1ec; --surface:#fffdf9; --sunk:#efeae1; --border:#e2dbcd;
    --ink:#26221c; --muted:#7c7466;
    --prop:#0d7268; --prop-soft:#0d726817;      /* Proposer — cool */
    --crit:#b4451f; --crit-soft:#b4451f17;       /* Critic — warm */
    --good:#3f7d3f; --good-soft:#3f7d3f1c; --bad:#b3261e; --bad-soft:#b3261e17;
    --serif:ui-serif,Georgia,"Iowan Old Style","Times New Roman",serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){ :root{
    --paper:#17140f; --surface:#201c15; --sunk:#120f0b; --border:#332d23;
    --ink:#ece5d6; --muted:#9a9081;
    --prop:#43b8a8; --prop-soft:#43b8a81f; --crit:#e2814a; --crit-soft:#e2814a1f;
    --good:#6cba6c; --good-soft:#6cba6c22; --bad:#f0857e; --bad-soft:#f0857e1f; } }
  :root[data-theme=light]{ --paper:#f4f1ec;--surface:#fffdf9;--sunk:#efeae1;--border:#e2dbcd;
    --ink:#26221c;--muted:#7c7466;--prop:#0d7268;--prop-soft:#0d726817;--crit:#b4451f;--crit-soft:#b4451f17;
    --good:#3f7d3f;--good-soft:#3f7d3f1c;--bad:#b3261e;--bad-soft:#b3261e17; }
  :root[data-theme=dark]{ --paper:#17140f;--surface:#201c15;--sunk:#120f0b;--border:#332d23;
    --ink:#ece5d6;--muted:#9a9081;--prop:#43b8a8;--prop-soft:#43b8a81f;--crit:#e2814a;--crit-soft:#e2814a1f;
    --good:#6cba6c;--good-soft:#6cba6c22;--bad:#f0857e;--bad-soft:#f0857e1f; }

  *{box-sizing:border-box} html,body{height:100%}
  body{margin:0;display:grid;grid-template-rows:auto 1fr;height:100vh;overflow:hidden;
       font:15px/1.6 var(--sans);background:var(--paper);color:var(--ink);
       -webkit-font-smoothing:antialiased}
  .mono{font-family:var(--mono)} .tnum{font-variant-numeric:tabular-nums}
  .muted{color:var(--muted)}
  .sec{font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);font-weight:600}
  button{font:inherit;color:inherit;cursor:pointer}
  kbd{font:10px var(--mono);border:1px solid var(--border);border-bottom-width:2px;border-radius:4px;
      padding:0 4px;color:var(--muted);background:var(--paper)}

  /* top bar — full width */
  #topbar{display:flex;align-items:center;gap:14px;padding:0 16px;height:52px;
          border-bottom:1.5px solid var(--ink);background:var(--surface)}
  .brand{display:flex;align-items:baseline;gap:9px}
  .brand h1{margin:0;font:600 18px/1 var(--serif);letter-spacing:.01em}
  .brand .tag{font:italic 12.5px var(--serif);color:var(--muted)}
  #crumb{font-size:12px}
  .spacer{flex:1}
  .search{display:flex;align-items:center;gap:6px;border:1px solid var(--border);border-radius:8px;
          padding:5px 9px;background:var(--paper)}
  .search input{border:0;background:transparent;color:var(--ink);font:13px var(--sans);width:150px;outline:none}
  .iconbtn{border:1px solid var(--border);border-radius:8px;background:var(--paper);padding:6px 10px;font-size:13px}
  .iconbtn:hover{background:var(--sunk)}

  /* three-column shell */
  /* graph column: one fixed width for every encoding (named pills need the wider of the
     two, and a column that resized per encoding made the whole page jump) */
  #shell{display:grid;grid-template-columns:288px minmax(0,1fr) 580px;min-height:0}
  body.railhidden #shell{grid-template-columns:minmax(0,1fr) 580px}
  body.railhidden #rail{display:none}

  /* left rail — browse */
  #rail{display:flex;flex-direction:column;min-height:0;border-right:1px solid var(--border);background:var(--surface)}
  #railhead{display:grid;gap:9px;padding:13px 16px;border-bottom:1px solid var(--border)}
  .chips{display:flex;gap:6px}
  .chip{border:1.5px solid var(--border);border-radius:20px;padding:2px 11px;font-size:12px;
        background:transparent;color:var(--muted)}
  .chip.on{border-color:var(--ink);color:var(--ink);font-weight:600}
  .chip.on[data-o=correct]{border-color:var(--good);color:var(--good)}
  .chip.on[data-o=wrong]{border-color:var(--bad);color:var(--bad)}
  .selrow{display:grid;grid-template-columns:1fr 1fr;gap:7px}
  select{width:100%;padding:6px 8px;border:1px solid var(--border);border-radius:8px;
       background:var(--paper);color:var(--ink);font:12.5px var(--sans)}
  select:focus{outline:2px solid var(--prop);outline-offset:-1px}
  #count{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase}
  #list{overflow-y:auto;min-height:0}
  .item{display:flex;gap:10px;align-items:center;padding:9px 16px;cursor:pointer;border-bottom:1px solid var(--border)}
  .item:hover{background:var(--sunk)} .item.sel{background:var(--prop-soft);box-shadow:inset 3px 0 0 var(--prop)}
  .item .id{flex:1;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .dot{width:8px;height:8px;border-radius:50%;flex:none}
  .dot.g{background:var(--good)} .dot.b{background:var(--bad)}

  /* center — reading column */
  #center{position:relative;display:grid;grid-template-rows:auto 1fr;min-height:0;min-width:0}
  #empty{grid-row:1/-1;display:grid;place-items:center;color:var(--muted);font:italic 16px var(--serif)}
  #head,#transcript{display:none}
  body.ready #empty{display:none}
  body.ready #head{display:block} body.ready #transcript{display:block}
  #head{padding:16px 30px 14px;border-bottom:1px solid var(--border);background:var(--surface)}
  #head .hrow{display:flex;align-items:center;gap:14px;margin:0 0 12px;flex-wrap:wrap}
  #head .idparts{display:flex;flex-wrap:wrap;gap:8px}
  #head .seg{display:inline-flex;align-items:baseline;gap:6px;padding:5px 12px;border:1px solid var(--border);
             border-radius:8px;background:var(--paper);font:14.5px var(--mono);color:var(--muted)}
  #head .seg i{font:9.5px/1 var(--sans);font-style:normal;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
  #head .seg.factor{background:var(--sunk);color:var(--ink);font-weight:600}
  .stamp{font:700 11px var(--sans);letter-spacing:.14em;text-transform:uppercase;padding:3px 10px;
         border:1.5px solid currentColor;border-radius:4px;transform:rotate(-1.5deg)}
  .stamp.good{color:var(--good)} .stamp.bad{color:var(--bad)}
  #head .qline{margin-left:auto;text-align:right;font:italic 21px var(--serif);color:var(--ink)}
  #head .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(124px,1fr));gap:10px;margin-top:13px}
  #head .stat{display:flex;align-items:center;gap:10px;padding:9px 12px;border:1px solid var(--border);
              border-radius:10px;background:var(--paper)}
  #head .stat .ic{color:var(--muted);flex:none}
  #head .stat .lab{font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
  #head .stat .val{font:600 15px var(--mono);color:var(--ink);margin-top:1px}
  #head .stat .val.good-fg{color:var(--good)} #head .stat .val.bad-fg{color:var(--bad)}
  #head .stat .sub{font:400 11px var(--mono);color:var(--muted)}
  #rawenc{padding:14px 16px;border-top:1px solid var(--border)}
  #rawenc summary{cursor:pointer;user-select:none;display:flex;align-items:center;gap:8px;list-style:none;
                  padding:5px 7px;margin:-5px -7px;border-radius:7px;transition:background .12s ease}
  #rawenc summary::-webkit-details-marker{display:none}
  #rawenc summary:hover{background:var(--sunk)}
  #rawenc .chev{margin-left:auto;color:var(--muted);display:block;transition:transform .18s ease}
  #rawenc[open] .chev{transform:rotate(90deg)}
  #rawenc pre{margin:10px 0 2px;padding:11px 12px;max-height:150px;overflow:auto;
              background:color-mix(in srgb, var(--ink) 8%, var(--surface));
              border:1px solid var(--border);border-radius:8px;
              white-space:pre-wrap;word-break:break-word;font:12px/1.6 var(--mono);color:var(--ink)}

  #transcript{overflow-y:auto;min-height:0;padding:24px 30px 64px}
  .turn{display:flex;gap:11px;margin:0 auto 18px;max-width:640px;align-items:flex-start;
        animation:rise .3s ease both}
  .turn.critic{flex-direction:row-reverse}
  @keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .av{width:34px;height:34px;flex:none;display:block}
  .av .disc{fill:currentColor}
  .av .em{fill:none;stroke:var(--surface);stroke-width:2.1;stroke-linecap:round;stroke-linejoin:round}
  .turn.proposer .av{color:var(--prop)} .turn.critic .av{color:var(--crit)}
  .bubble{min-width:0;max-width:560px}
  .who{display:flex;align-items:baseline;gap:9px;margin-bottom:6px}
  .turn.critic .who{flex-direction:row-reverse}
  .who .name{font:600 14px var(--serif);letter-spacing:.02em}
  .turn.proposer .name{color:var(--prop)} .turn.critic .name{color:var(--crit)}
  .who .role{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
  .who .tok{font:11px var(--mono);color:var(--muted)}
  .say{margin:0;padding:11px 14px;border-radius:12px;border:1px solid var(--border);background:var(--surface);
       white-space:pre-wrap;word-break:break-word;font:13px/1.55 var(--mono);color:var(--ink)}
  .turn.proposer .say{border-bottom-left-radius:3px;border-left:2.5px solid var(--prop)}
  .turn.critic .say{border-bottom-right-radius:3px;border-right:2.5px solid var(--crit);background:var(--crit-soft)}
  .ans{margin-top:7px;font:13px var(--sans);color:var(--muted)}
  .ans b{color:var(--ink)} .ans .chg{color:var(--crit)}
  .verdict{font:700 10.5px var(--sans);letter-spacing:.09em;text-transform:uppercase;padding:2px 9px;border-radius:20px}
  .verdict.agree{background:var(--good-soft);color:var(--good)} .verdict.revise{background:var(--crit-soft);color:var(--crit)}
  .unpar{font:10px var(--mono);color:var(--muted);margin-left:6px}

  /* right — graph column */
  #graphcol{display:flex;flex-direction:column;min-height:0;border-left:1px solid var(--border);background:var(--sunk)}
  .gcolhead{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;border-bottom:1px solid var(--border)}
  #gempty{flex:1;display:grid;place-items:center;color:var(--muted);font:italic 13px var(--serif)}
  #gwrap{display:none;flex-direction:column;flex:1;min-height:0;overflow-y:auto}
  body.ready #gempty{display:none} body.ready #gwrap{display:flex}
  #cyframe{flex:1;min-height:150px;margin:12px;border:1px solid var(--border);border-radius:12px;
           background:var(--surface);overflow:hidden}
  #cy{width:100%;height:100%}
  #legend{display:flex;gap:14px;padding:0 16px;font-size:11.5px;color:var(--muted);flex-wrap:wrap}
  #legend span{display:flex;align-items:center;gap:6px}
  .sw{width:11px;height:11px;border-radius:50%;display:inline-block}
  .sw.q{background:var(--prop)} .sw.o{border:1.5px solid var(--muted)}
  .panel{padding:14px 16px;border-top:1px solid var(--border);margin-top:12px}
  #bars{display:flex;gap:4px;align-items:flex-end;height:46px;margin:8px 0 5px}
  .bar{flex:1;border-radius:3px 3px 0 0;min-height:4px}
  .bar.proposer{background:var(--prop)} .bar.critic{background:var(--crit)}
  #costlab{font:10.5px var(--mono);color:var(--muted)}

  :focus-visible{outline:2px solid var(--prop);outline-offset:2px}
  @media (prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body>
<div id="topbar">
  <button id="railtoggle" class="iconbtn" title="toggle browse list" aria-label="toggle browse list">&#9776;</button>
  <div class="brand"><h1>Debate Reader</h1><span class="tag">Proposer v. Critic</span></div>
  <span id="crumb" class="mono muted"></span>
  <span class="spacer"></span>
  <label class="search">&#9906;<input id="q" placeholder="search id…" aria-label="search instance id"><kbd>/</kbd></label>
  <button id="themebtn" class="iconbtn" title="toggle light / dark" aria-label="toggle theme">&#9681;</button>
</div>
<div id="shell">
  <aside id="rail">
    <div id="railhead">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="sec">Browse</span><span id="count" class="muted mono"></span>
      </div>
      <div class="chips" id="chips">
        <button class="chip on" data-o="all">all</button>
        <button class="chip" data-o="correct">correct</button>
        <button class="chip" data-o="wrong">wrong</button>
      </div>
      <div class="selrow"><select id="fTask" aria-label="task"></select><select id="fEnc" aria-label="encoding"></select></div>
    </div>
    <div id="list"></div>
  </aside>
  <main id="center">
    <div id="empty">Pick a debate to read.</div>
    <header id="head"></header>
    <section id="transcript"></section>
  </main>
  <aside id="graphcol">
    <div class="gcolhead"><span class="sec">Graph</span><span id="gmeta" class="mono muted"></span></div>
    <div id="gempty">no debate open</div>
    <div id="gwrap">
      <div id="cyframe"><div id="cy"></div></div>
      <div id="legend"><span><span class="sw q"></span><span id="legq">query node</span></span><span><span class="sw o"></span>other nodes</span></div>
      <div class="panel"><span class="sec">Cost</span><div id="bars"></div><div id="costlab"></div></div>
      <details id="rawenc" open><summary><span class="sec">Raw encoding</span><svg class="chev" viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 4l4 4-4 4"/></svg></summary><pre id="enc"></pre></details>
    </div>
  </aside>
</div>
<script>
let INDEX=[], sel=null, cy=null, outcome='all', LASTG=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const show=v=>esc(JSON.stringify(v));
const eq=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// Illustrated agent avatars (inline SVG, theme-adaptive): a lightbulb for the
// Proposer (who has the idea), a magnifier for the Critic (who examines it).
const AV_PROP=`<svg class="av" viewBox="0 0 40 40" aria-hidden="true"><circle class="disc" cx="20" cy="20" r="20"/>`
  +`<g class="em"><circle cx="20" cy="17" r="7"/><path d="M16.6 26.5h6.8M17.6 29.5h4.8"/><path d="M20 13.6v3.4M17.9 18.6h4.2"/></g></svg>`;
const AV_CRIT=`<svg class="av" viewBox="0 0 40 40" aria-hidden="true"><circle class="disc" cx="20" cy="20" r="20"/>`
  +`<g class="em"><circle cx="18" cy="18" r="6.4"/><path d="M22.9 22.9 27.6 27.6"/></g></svg>`;
// The header names the queried node(s) the way the encoding does ("degree of Robert?"),
// so it reads like the question the model was actually asked. The verbatim `Q:` line
// stays one glance away in the Raw encoding panel.
const labOf=(g,n)=>(g&&g.labels&&g.labels[''+n])||''+n;
const QLABEL={node_degree:n=>`degree of ${n[0]}?`, connected_nodes:n=>`neighbors of ${n[0]}?`,
              edge_existence:n=>`${n[0]} and ${n[1]} connected?`};

function opts(sel,vals){ const lab=sel.options[0]?.text||'all';
  sel.innerHTML=`<option value="all">${lab}</option>`+vals.map(v=>`<option value="${v}">${v}</option>`).join(''); }
function filtered(){ const q=$('q').value.trim(),t=$('fTask').value,e=$('fEnc').value;
  return INDEX.filter(d=>(!q||d.instance_id.includes(q))&&(t==='all'||d.task===t)&&(e==='all'||d.encoding===e)
    &&(outcome==='all'||(outcome==='correct')===d.correct)); }
function renderList(){ const it=filtered(); $('count').textContent=it.length+' / '+INDEX.length;
  const L=$('list'); L.innerHTML='';
  it.forEach(d=>{ const el=document.createElement('div'); el.className='item'+(sel===d.instance_id?' sel':'');
    el.tabIndex=0; el.innerHTML=`<span class="dot ${d.correct?'g':'b'}"></span>`
      +`<span class="id mono">${d.instance_id}</span>`;
    const go=()=>{sel=d.instance_id;renderList();openDebate(d.instance_id);};
    el.onclick=go; el.onkeydown=ev=>{if(ev.key==='Enter')go();}; L.appendChild(el); }); }

async function openDebate(id){
  const d=await(await fetch('/api/trace?id='+encodeURIComponent(id))).json();
  document.body.classList.add('ready');
  const ok=d.correct, total=d.n_prompt_tokens+d.n_gen_tokens;
  const qn=((d.graph&&d.graph.query_nodes)||[]).map(n=>labOf(d.graph,n));
  const qt=(QLABEL[d.task]&&qn.length>=(d.task==='edge_existence'?2:1))
    ? QLABEL[d.task](qn) : d.task.replace(/_/g,' ');   // no graph data (no manifest): task name
  $('crumb').textContent='/ '+d.task+' · '+d.encoding;
  const okc=ok?'good-fg':'bad-fg';
  const svg=p=>`<svg class="ic" width="19" height="19" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
  const IC_A=svg('<rect x="2" y="3" width="12" height="8" rx="2"/><path d="M5.5 11.5v2.3l2.6-2.3"/>');    // answer — speech bubble
  const IC_T=svg('<path d="M4 2.4v11.2"/><path d="M4 3.1h7.2l-1.7 2.3 1.7 2.3H4"/>');                     // truth — flag
  const IC_R=svg('<path d="M3 6.2h9.6"/><path d="M10.6 4.2 12.8 6.2 10.6 8.2"/><path d="M13 10.2H3.4"/><path d="M5.4 8.2 3.2 10.2 5.4 12.2"/>'); // turns — exchange
  const IC_K=svg('<path d="M8 2v12"/><path d="M10.6 5.1C10.6 3.9 9.4 3.2 8 3.2 6.5 3.2 5.3 4 5.3 5.4 5.3 8 10.7 7.2 10.7 10 10.7 11.5 9.4 12.4 8 12.4 6.6 12.4 5.4 11.7 5.4 10.4"/>'); // tokens — dollar
  const stat=(ic,lab,val,cls,sub)=>`<div class="stat">${ic}<div><div class="lab">${lab}</div>`
    +`<div class="val ${cls||''}">${val}${sub?` <span class="sub">${sub}</span>`:''}</div></div></div>`;
  const p=d.instance_id.split('/'), seg=(lab,val,cls)=>`<span class="seg ${cls||''}"><i>${lab}</i>${esc(val)}</span>`;
  $('head').innerHTML=`<div class="hrow"><div class="idparts">`
    +seg('seed',p[0]||'?')+seg('graph',p[1]||'?')+seg('task',d.task,'factor')+seg('encoding',d.encoding,'factor')
    +`</div><span class="stamp ${ok?'good':'bad'}">${ok?'correct':'wrong'}</span>`
    +`<div class="qline">&ldquo;${esc(qt)}&rdquo;</div></div>`
    +`<div class="stats">`
    +stat(IC_A,'Answer',show(d.parsed_answer),okc)
    +stat(IC_T,'Truth',show(d.ground_truth))
    +stat(IC_R,'Turns',d.n_responses)
    +stat(IC_K,'Tokens',total.toLocaleString(),'',d.n_prompt_tokens.toLocaleString()+' read')
    +`</div>`;
  const turns=d.turns||[];
  const T=$('transcript'); T.scrollTop=0; let prev=undefined;
  T.innerHTML=turns.map((t,i)=>{ const isC=t.role==='critic';
    let foot='';
    if(!isC){ const same=prev!==undefined&&eq(t.parsed,prev);
      const note=same?' <span class="muted mono">(unchanged)</span>':(prev!==undefined?' <span class="chg mono">(changed)</span>':'');
      foot=`<div class="ans">answer &rarr; <b class="mono">${show(t.parsed)}</b>${note}</div>`; prev=t.parsed; }
    let v=''; if(isC&&t.verdict){ v=`<span class="verdict ${t.verdict==='AGREE'?'agree':'revise'}">${t.verdict}</span>`
      +(t.critic_verdict_parsed===false?'<span class="unpar">unparsed → default</span>':''); }
    const name=isC?'Critic':'Proposer', role=i===0?'opens':(isC?'reviews':'revises');
    return `<div class="turn ${t.role}" style="animation-delay:${i*45}ms">`
      +`<div class="av-wrap">${isC?AV_CRIT:AV_PROP}</div>`
      +`<div class="bubble"><div class="who"><span class="name">${name}</span>`
      +`<span class="role">${role}</span>${v}<span class="tok tnum">${(t.n_prompt_tokens||0)+(t.n_gen_tokens||0)} tok</span></div>`
      +`<div class="say">${esc(t.raw||'')}</div>${foot}</div></div>`; }).join('');
  renderGraph(d.graph,d.task);
  renderCost(turns,total);
}

function renderCost(turns,total){
  const tot=t=>(t.n_prompt_tokens||0)+(t.n_gen_tokens||0);
  const max=Math.max(1,...turns.map(tot));
  $('bars').innerHTML=turns.map(t=>`<div class="bar ${t.role}" style="height:${Math.max(8,Math.round(tot(t)/max*100))}%"`
    +` title="turn ${t.role}: ${tot(t)} tok"></div>`).join('');
  $('costlab').textContent=`tokens per turn · ${total.toLocaleString()} total`;
}
// Nodes are drawn with the names the model actually read (server-supplied `labels`):
// integers for adjacency/incident, people for friendship. The integer id stays available
// on hover, so a payload id can still be matched to what is on screen.
function renderGraph(g,task){
  LASTG={g,task};
  const lab=n=>labOf(g,n);
  // Only the first queried node is lit for now; edge_existence's second endpoint joins it
  // when the per-task drawings land.
  const q=(g&&g.query_nodes&&g.query_nodes.length)?g.query_nodes[0]:null;
  $('enc').textContent=g?g.encoding_text:'';
  $('gmeta').textContent=g?`${g.nnodes} nodes · ${g.edges.length} edges`:'';
  $('legq').textContent=q!=null?`query node ${lab(q)}`:'query node';
  if(cy){cy.destroy();cy=null;} if(!g||typeof cytoscape==='undefined'){ return; }
  const prop=cssv('--prop'),ink=cssv('--ink'),muted=cssv('--muted'),surf=cssv('--surface'),paper=cssv('--paper');
  // Named encodings get label-sized pills at 13px; numbered ones keep the original discs.
  const nodeSize=g.named?{'width':'label','height':'label','padding':'9px','font-size':13}
                        :{'width':27,'height':27,'font-size':12};
  const qSize=g.named?{'padding':'12px'}:{'width':32,'height':32};
  const inc=new Set(); g.edges.forEach(([a,b])=>{ if(a===q||b===q) inc.add(a+'-'+b); });
  const els=g.nodes.map(n=>({data:{id:''+n,label:lab(n)},position:g.positions[''+n],classes:n===q?'q':''}))
    .concat(g.edges.map(([a,b])=>({data:{id:a+'-'+b,source:''+a,target:''+b},classes:inc.has(a+'-'+b)?'inc':''})));
  cy=cytoscape({container:$('cy'),elements:els,layout:{name:'preset'},
    style:[
      {selector:'node',style:{'label':'data(label)','text-wrap':'none','background-color':surf,
        'border-width':1.5,'border-color':muted,'color':ink,'font-family':'monospace',
        'text-valign':'center','text-halign':'center',...nodeSize}},
      {selector:'node.q',style:{'background-color':prop,'border-color':prop,'color':paper,
        'font-weight':'bold',...qSize}},
      {selector:'edge',style:{'width':1.5,'line-color':muted,'opacity':.45,'curve-style':'straight'}},
      {selector:'edge.inc',style:{'line-color':prop,'opacity':1,'width':2.5}}]});
  cy.on('mouseover','node',e=>{ $('cy').title='node '+e.target.id(); });
  cy.on('mouseout','node',()=>{ $('cy').title=''; });
  cy.fit(undefined,26);
}

// filters
['q','fTask','fEnc'].forEach(id=>{ $(id).oninput=renderList; $(id).onchange=renderList; });
$('chips').querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
  outcome=c.dataset.o; $('chips').querySelectorAll('.chip').forEach(x=>x.classList.toggle('on',x===c)); renderList(); });
$('fTask').innerHTML='<option value="all">every task</option>'; $('fEnc').innerHTML='<option value="all">every encoding</option>';

// chrome: rail collapse + theme toggle (re-styles the graph so it tracks the theme)
$('railtoggle').onclick=()=>document.body.classList.toggle('railhidden');
let theme=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
document.documentElement.dataset.theme=theme;
$('themebtn').onclick=()=>{ theme=theme==='dark'?'light':'dark'; document.documentElement.dataset.theme=theme;
  if(LASTG) renderGraph(LASTG.g,LASTG.task); };

// keyboard: / search · j/k or n/p walk the queue · r toggles raw encoding
function step(dir){ const it=filtered(); if(!it.length)return;
  let i=it.findIndex(d=>d.instance_id===sel);
  i=i<0?(dir>0?0:it.length-1):Math.min(it.length-1,Math.max(0,i+dir));
  const d=it[i]; sel=d.instance_id; renderList(); openDebate(d.instance_id);
  const node=$('list').querySelector('.item.sel'); if(node) node.scrollIntoView({block:'nearest'}); }
document.addEventListener('keydown',e=>{
  const typing=/^(input|select|textarea)$/i.test(e.target.tagName);
  if(e.key==='/'&&!typing){ e.preventDefault(); $('q').focus(); return; }
  if(e.key==='Escape'&&typing){ e.target.blur(); return; }
  if(typing) return;
  if(e.key==='r'){ const el=$('rawenc'); if(el) el.open=!el.open; return; }
  if(['j','n','ArrowDown'].includes(e.key)){ e.preventDefault(); step(1); }
  if(['k','p','ArrowUp'].includes(e.key)){ e.preventDefault(); step(-1); } });

fetch('/api/index').then(r=>r.json()).then(idx=>{ INDEX=idx;
  opts($('fTask'),[...new Set(idx.map(d=>d.task))]); opts($('fEnc'),[...new Set(idx.map(d=>d.encoding))]);
  renderList(); });
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a run dir containing debate/ rows + trace sidecars")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    rows, turns, instances, index = load(args.run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(rows, turns, instances, index))
    print(f"{len(index)} debates -> http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
