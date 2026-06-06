#!/usr/bin/env python3
"""Distillation-Dataset: lokale Code-Snippets × Goal → mistral dry_run categorize
(volle mixed Methode gegen Prod-Codebook) → SFT-Targets {paraphrase,generalization,label}.
Lehrt qwen3.5:2b die richtige Granularität + Dedup-Verhalten von mistral."""
import ast, json, os, sys, urllib.request, glob, random

TOKEN = open(os.path.expanduser("~/.config/mayring/hook.jwt")).read().strip()
URL = "https://mcp.linn.games/codebooks/categorize"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
REPOS = ["/home/nileneb/Desktop/MayringCoder/src", "/home/nileneb/Desktop/MayringCoder/vendor/mayring-core/mayring_core"]

def snippets():
    out = []
    files = []
    for r in REPOS:
        files += glob.glob(f"{r}/**/*.py", recursive=True)
    random.seed(7); random.shuffle(files)
    for f in files:
        try: src = open(f).read()
        except Exception: continue
        try: tree = ast.parse(src)
        except Exception: continue
        mod_doc = (ast.get_docstring(tree) or "").strip().split("\n")[0]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                try: seg = ast.get_source_segment(src, node)
                except Exception: seg = None
                if not seg or len(seg) < 80 or len(seg) > 1400: continue
                doc = (ast.get_docstring(node) or "").strip().split("\n")[0]
                # Goal = Modul-Zweck + Symbol (Selektionskriterium)
                goal = (mod_doc or doc or f"{os.path.basename(f)}: {node.name}")[:120]
                out.append({"goal": goal, "text": seg, "src": f"{os.path.basename(f)}:{node.name}"})
                if len(out) >= N * 3: return out
    return out

def categorize(text, goal, model):
    body = json.dumps({"text": text, "task": goal, "model": model, "dry_run": True}).encode()
    req = urllib.request.Request(URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r: return json.load(r)

cands = snippets()
random.shuffle(cands)
records, used = [], 0
for c in cands:
    if used >= N: break
    try: res = categorize(c["text"], c["goal"], "mistral:7b-instruct")
    except Exception as e: print(f"  skip ({e})"); continue
    if res.get("error") or not res.get("label"): print(f"  skip (no label: {res.get('error','')})"); continue
    # SFT-Record (messages-Format, Qwen-Chat): system+user=Prompt, assistant=JSON-Target
    target = json.dumps({"paraphrase": res["paraphrase"], "generalization": res["generalize"],
                         "label": res["label"]}, ensure_ascii=False)
    records.append({"goal": c["goal"], "text": c["text"], "src": c["src"],
                    "target": target, "match": res["match"]})
    used += 1
    print(f"[{used}/{N}] {c['src'][:45]} -> {res['label']} ({res['match']})")

json.dump(records, open("/tmp/distill_full.json", "w"), indent=2, ensure_ascii=False)
print(f"\nSaved {len(records)} records -> /tmp/distill_full.json")
