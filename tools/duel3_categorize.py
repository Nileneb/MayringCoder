import json, urllib.request, os
TOKEN = open(os.path.expanduser("~/.config/mayring/hook.jwt")).read().strip()
PAIRS = json.load(open("/tmp/duel_eval.json"))
MODELS = ["mistral:7b-instruct", "qwen3.5:2b", "qwen3.5-mayring:2b"]
URL = "https://mcp.linn.games/codebooks/categorize"
def call(text, task, model):
    body=json.dumps({"text":text,"task":task,"model":model,"dry_run":True}).encode()
    req=urllib.request.Request(URL,data=body,method="POST",headers={"Authorization":f"Bearer {TOKEN}","Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=120) as r: return json.load(r)
    except Exception as e: return {"error":str(e)}
results=[]
for i,p in enumerate(PAIRS):
    row={"i":i+1,"goal":p.get("goal",""),"text":p.get("text","")[:200]}
    for m in MODELS:
        out=call(p.get("text",""),p.get("goal",""),m)
        row[m]={k:out.get(k) for k in ("label","match","paraphrase","generalize","error")}
        print(f"[{i+1}] {m:24} -> {out.get('label')} ({out.get('match')})" + (f" ERR:{out.get('error')}" if out.get('error') else ''))
    results.append(row)
json.dump(results,open("/tmp/duel3_results.json","w"),indent=2,ensure_ascii=False)
print("\nSaved /tmp/duel3_results.json")
