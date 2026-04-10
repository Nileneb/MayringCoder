#!/bin/bash
# Full Annotation + QA + Benchmark Pipeline
# Läuft autonom im Hintergrund, Ergebnis in cache/pipeline_report.txt

set -e
cd /home/nileneb/Desktop/MayringCoder
VENV=".venv/bin/python"
REPORT="cache/pipeline_report.txt"
ANNOTATED="cache/training_annotated.jsonl"

echo "=== Pipeline gestartet: $(date) ===" > "$REPORT"

# --- Step 1: Full Annotation (1243 Samples, ~1h) ---
echo "[Step 1] Starte Batch-Annotation (1243 Samples)..." | tee -a "$REPORT"
rm -f "$ANNOTATED"
$VENV tools/annotate_training_data.py \
    --model qwen3.5:9b \
    --delay 2.0 \
    --ollama-url http://localhost:11434 \
    2>&1 | tee -a "$REPORT"

echo "" >> "$REPORT"

# --- Step 2: Quality Check (Stichprobe) ---
echo "[Step 2] Qualitätskontrolle — Stichprobe 30 Samples..." | tee -a "$REPORT"
$VENV -c "
import json, random
from pathlib import Path

random.seed(42)
lines = Path('$ANNOTATED').read_text().splitlines()
if not lines:
    print('ERROR: Keine annotierten Samples gefunden')
    exit(1)

samples = [json.loads(l) for l in lines if 'annotation' in l]
sample_size = min(30, len(samples))
stichprobe = random.sample(samples, sample_size)

# Stats
qualities = {}
precisions = []
for s in samples:
    a = s.get('annotation', {})
    q = a.get('overall_quality', '?')
    qualities[q] = qualities.get(q, 0) + 1
    p = a.get('precision')
    if isinstance(p, (int, float)):
        precisions.append(p)

print(f'Gesamt annotiert: {len(samples)}')
print(f'Qualitätsverteilung:')
for q, c in sorted(qualities.items(), key=lambda x: -x[1]):
    print(f'  {q}: {c} ({100*c/len(samples):.0f}%)')

if precisions:
    avg_p = sum(precisions) / len(precisions)
    print(f'Durchschnittliche Precision: {avg_p:.2f}')

# Stichprobe Details
print(f'\nStichprobe ({sample_size} Samples):')
for s in stichprobe[:10]:
    a = s.get('annotation', {})
    label = s.get('label', '?')[:25]
    q = a.get('overall_quality', '?')
    tp = a.get('true_positives', '?')
    fp = a.get('false_positives', '?')
    reason = a.get('reasoning', '')[:80]
    print(f'  [{q}] {label} — TP:{tp} FP:{fp} — {reason}')

# Export good-only annotated
good = [s for s in samples if s.get('annotation',{}).get('overall_quality') == 'good']
good_path = Path('cache/training_annotated_good.jsonl')
with good_path.open('w') as f:
    for s in good:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f'\nGood-only Export: {len(good)} Samples → {good_path}')

# Export für Fine-Tuning (instruction format)
ft_path = Path('cache/training_finetuning_ready.jsonl')
count = 0
with ft_path.open('w') as f:
    for s in samples:
        a = s.get('annotation', {})
        if a.get('overall_quality') in ('good', 'partial') and a.get('precision', 0) >= 0.5:
            f.write(json.dumps({
                'instruction': 'Analyze this code for potential issues. Return structured JSON.',
                'input': s.get('prompt', '')[:3000],
                'output': s.get('raw_response', '')[:3000],
                'model': s.get('model', ''),
                'quality': a.get('overall_quality'),
                'precision': a.get('precision', 0),
            }, ensure_ascii=False) + '\n')
            count += 1
print(f'Fine-Tuning-Ready Export: {count} Samples → {ft_path}')
" 2>&1 | tee -a "$REPORT"

echo "" >> "$REPORT"

# --- Step 3: Benchmark (mit allen Daten) ---
echo "[Step 3] Retrieval-Benchmark (40 Queries)..." | tee -a "$REPORT"
$VENV src/benchmark_retrieval.py \
    --queries benchmarks/retrieval_queries.yaml \
    --top-k 5 \
    2>&1 | tee -a "$REPORT"

echo "" >> "$REPORT"

# --- Step 4: Zusammenfassung ---
echo "[Step 4] Zusammenfassung..." | tee -a "$REPORT"
$VENV -c "
import sqlite3
conn = sqlite3.connect('cache/memory.db')
print('=== Memory DB Final Stats ===')
print(f'Sources: {conn.execute(\"SELECT COUNT(*) FROM sources\").fetchone()[0]}')
print(f'Active Chunks: {conn.execute(\"SELECT COUNT(*) FROM chunks WHERE is_active=1\").fetchone()[0]}')
for row in conn.execute('''
    SELECT s.source_type, COUNT(c.chunk_id)
    FROM chunks c JOIN sources s ON c.source_id = s.source_id
    WHERE c.is_active = 1 GROUP BY s.source_type ORDER BY COUNT(c.chunk_id) DESC
''').fetchall():
    print(f'  {row[0]}: {row[1]}')
" 2>&1 | tee -a "$REPORT"

echo "" >> "$REPORT"
echo "=== Pipeline fertig: $(date) ===" | tee -a "$REPORT"
echo ""
echo "Ergebnis: $REPORT"
echo "Datasets:"
ls -lh cache/training_annotated*.jsonl cache/training_finetuning_ready.jsonl 2>/dev/null
