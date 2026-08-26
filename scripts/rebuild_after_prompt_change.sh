#!/usr/bin/env bash
# Full rebuild chain after a system-prompt change — runs unattended to completion.
#
#   nohup bash scripts/rebuild_after_prompt_change.sh > /dev/null 2>&1 &
#
# Steps, in order (each blocks on the previous):
#   0. wait for any generation already in flight
#   1. regenerate the 50 study briefings (risk verdicts PINNED) + a 25-question
#      held-out set that was never used by the study, for overfitting control
#   2. verify both sets against the prompt's intent
#   3. re-freeze assignments, rebuild P1 packets and both psychologist workbooks
#   4. swap the GPU to the RAGAS judge (Qwen) and re-run RAGAS
#   5. swap back to the generator (Gemma) and mint fresh counselor share links
#   6. write a human-readable report
#
# Everything lands in outputs/logs/rebuild_<stamp>.{md,log}. The .md is the one
# to read. On failure the script stops and the report says where.
#
# Safe to run after a disconnect: it is detached from the terminal, and every
# destructive step is preceded by the backups named in BASELINE / a fresh one.
#
# Env overrides:
#   BASELINE=<dir>   backup whose risk verdicts are pinned (default: pre-session)
#   SKIP_RAGAS=1     stop after step 3 (no GPU swap, chatbot stays up)

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export HF_HOME="$ROOT/.hf_cache"
export GRADIO_ANALYTICS_ENABLED=False
export PYTHONUNBUFFERED=1

BASELINE="${BASELINE:-archive/pre_prompt_fix_20260801-115847}"
SKIP_RAGAS="${SKIP_RAGAS:-0}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p outputs/logs
LOG="$ROOT/outputs/logs/rebuild_${STAMP}.log"
REPORT="$ROOT/outputs/logs/rebuild_${STAMP}.md"

say()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
add()  { printf '%s\n' "$*" >> "$REPORT"; }
fail() { say "GAGAL: $*"; add ""; add "## ⛔ BERHENTI"; add ""; add "Gagal pada langkah: **$***"; \
         add ""; add "Log lengkap: \`${LOG#$ROOT/}\`"; exit 1; }

add "# Rebuild after prompt change — ${STAMP}"
add ""
add "Baseline (pin risiko): \`$BASELINE\`"
add ""

# ---------------------------------------------------------------- helpers ---
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0; }

gpu_free() {
  # NEVER `pkill -f vllm` — the pattern matches this script's own command line.
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -n "$pids" ] && kill $pids 2>/dev/null
  for _ in $(seq 1 60); do
    [ "$(gpu_used)" -lt 2000 ] && { say "GPU bebas"; return 0; }
    sleep 5
  done
  return 1
}

serve() {   # $1 = model id
  say "menjalankan vLLM: $1"
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    nohup .venv-vllm/bin/vllm serve "$1" \
      --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 16384 \
      --port 8000 >> "$LOG" 2>&1 &
  disown
  # Wait for the requested model specifically. Checking only that :8000 answers is
  # a trap: the previous server keeps replying for a few seconds after its GPU
  # process is killed, so the loop would exit before the new model has loaded.
  for _ in $(seq 1 90); do
    if curl -sf --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -qF "\"$1\""; then
      say "vLLM siap ($1)"; return 0
    fi
    sleep 10
  done
  return 1
}

lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

# --------------------------------------------- 0. wait for in-flight work ---
say "== 0. menunggu generasi yang mungkin sedang berjalan =="
for _ in $(seq 1 360); do            # up to 60 min
  pgrep -f "generate_study_answers.py" >/dev/null 2>&1 || break
  sleep 10
done
pgrep -f "generate_study_answers.py" >/dev/null 2>&1 && fail "generasi lama tidak selesai dalam 60 menit"
say "tidak ada generasi berjalan"

# ------------------------------------------------------- 1. (re)generate ---
say "== 1. generasi =="
[ -f "$BASELINE/study_answers_chatbot.jsonl" ] || fail "baseline tidak ditemukan: $BASELINE"

if [ "$(lines outputs/analysis/study_answers_chatbot.jsonl)" -ne 50 ]; then
  say "regenerasi 50 briefing studi (risiko di-pin)"
  .venv-chat/bin/python scripts/generate_study_answers.py \
    --pin-risk-from "$BASELINE/study_answers_chatbot.jsonl" >> "$LOG" 2>&1 \
    || fail "regenerasi set studi"
else
  say "set studi sudah lengkap (50) — dilewati"
fi

if [ ! -f data/derived/questions_heldout25.jsonl ]; then
  say "membuat set held-out 25 pertanyaan"
  python - >> "$LOG" 2>&1 <<'PY' || fail "membuat held-out"
import json, random
full=[json.loads(l) for l in open("data/derived/questions_alodokter.jsonl")]
used={json.loads(l)["question_id"] for l in open("data/derived/questions_alodokter_sample50.jsonl")}
pool=[q for q in full if q["question_id"] not in used]
s=random.Random(20260628).sample(pool,25)
with open("data/derived/questions_heldout25.jsonl","w",encoding="utf-8") as fh:
    for i,q in enumerate(s,1):
        q["study_id"]=f"H{i:02d}"; fh.write(json.dumps(q,ensure_ascii=False)+"\n")
PY
fi

if [ "$(lines outputs/analysis/heldout25_chatbot.jsonl)" -ne 25 ]; then
  say "generasi 25 held-out (tanpa pin — belum pernah di-screen)"
  .venv-chat/bin/python scripts/generate_study_answers.py \
    --questions data/derived/questions_heldout25.jsonl \
    --out outputs/analysis/heldout25_chatbot.jsonl >> "$LOG" 2>&1 \
    || fail "generasi held-out"
else
  say "held-out sudah lengkap (25) — dilewati"
fi

add "## 1. Generasi"
add ""
add "- set studi  : $(lines outputs/analysis/study_answers_chatbot.jsonl) / 50"
add "- held-out   : $(lines outputs/analysis/heldout25_chatbot.jsonl) / 25"
add ""

# ------------------------------------------------------------ 2. verify ----
say "== 2. verifikasi =="
python - >> "$REPORT" 2>>"$LOG" <<'PY' || fail "verifikasi"
import json, re
MI      = re.compile(r"MI\.?\s?\d")
NONNUM  = re.compile(r"\[(?!\d+\])[^\]\n]{1,25}\]")
ROUTE   = re.compile(r"\bIGD\b|puskesmas terdekat|gawat darurat", re.I)
SUICIDE = re.compile(r"bunuh diri|mengakhiri hidup|melukai diri|menyakiti diri", re.I)
K5      = re.compile(r"alat bantu|bukan pengganti|tidak menggantikan", re.I)
BLAME   = re.compile(r"[Pp]edoman[^.\n]{0,50}tidak (?:men)?(?:cakup|mencakup|merinci)")
ROUTE_S = re.compile(r"[^.\n]*\b(?:IGD|puskesmas terdekat)\b[^.\n]*\.", re.I)

def audit(path, label):
    rows=[json.loads(l) for l in open(path)]
    n=len(rows)
    cited=uncited=0
    for r in rows:
        for m in ROUTE_S.finditer(r["generated_answer"]):
            if re.search(r"\[\d\]", m.group(0)): cited+=1
            else: uncited+=1
    raised=[r for r in rows if SUICIDE.search(r["generated_answer"])]
    noroute=[r for r in raised if not ROUTE.search(r["generated_answer"])]
    risky=[r for r in rows if r.get("risky")]
    risky_route=[r for r in risky if ROUTE.search(r["generated_answer"])]
    print(f"### {label}  (n={n})")
    print()
    print("| cek | hasil | target |")
    print("|---|--:|--:|")
    print(f"| sitasi [n] pada kalimat rute | **{cited}** | 0 |")
    print(f"| kalimat rute tanpa sitasi | {uncited} | — |")
    print(f"| kode modul pedoman | **{sum(len(MI.findall(r['generated_answer'])) for r in rows)}** | 0 |")
    print(f"| kurung siku non-numerik | {sum(len(NONNUM.findall(r['generated_answer'])) for r in rows)} | rendah |")
    print(f"| menyalahkan pedoman | {sum(len(BLAME.findall(r['generated_answer'])) for r in rows)} | rendah |")
    print(f"| mengangkat risiko TANPA rute | **{len(noroute)}** | 0 |")
    print(f"| kasus berisiko dgn rute | {len(risky_route)} / {len(risky)} | semua |")
    print(f"| memuat kalimat batas peran | {sum(1 for r in rows if K5.search(r['generated_answer']))} / {n} | hampir semua |")
    print()
    return cited, len(noroute)

print("## 2. Verifikasi")
print()
c1,n1 = audit("outputs/analysis/study_answers_chatbot.jsonl", "Set studi (50) — dipakai untuk menyetel prompt")
c2,n2 = audit("outputs/analysis/heldout25_chatbot.jsonl", "Held-out (25) — TIDAK pernah dilihat saat menyetel")
print("> **Kontrol overfitting.** Held-out diambil acak (seed 20260628) dari 323")
print("> pertanyaan korpus yang tidak dipakai studi. Bila angkanya sebanding dengan")
print("> set studi, perbaikan prompt bersifat struktural — bukan penyetelan ke 50 kasus.")
print()
if c1 or c2:
    print(f"> ⚠️ Masih ada sitasi pada kalimat rute (studi {c1}, held-out {c2}) — aturan 6 belum sepenuhnya dipatuhi.")
    print()
PY
say "verifikasi ditulis ke laporan"

# ----------------------------------------------- 3. re-freeze and rebuild ---
say "== 3. bekukan ulang + bangun ulang =="
python scripts/freeze_eval_assignments.py --force >> "$LOG" 2>&1 || fail "freeze_eval_assignments"
python scripts/build_eval_packets.py            >> "$LOG" 2>&1 || fail "build_eval_packets"
.venv/bin/python scripts/build_psychologist_workbooks.py >> "$LOG" 2>&1 || fail "build_psychologist_workbooks"

add "## 3. Pembekuan & pembangunan ulang"
add ""
python - >> "$REPORT" 2>>"$LOG" <<PY
import json
BK="$BASELINE"
o=json.load(open(f"{BK}/eval_case_assignments.json")); n=json.load(open("data/derived/eval_case_assignments.json"))
K=["study_id","question_id","block","risky","in_counselor_sample","counselor_condition","in_p1","p1_rater","p1_form"]
oc={c["study_id"]:{k:c[k] for k in K} for c in o["cases"]}
nc={c["study_id"]:{k:c[k] for k in K} for c in n["cases"]}
d=[s for s in oc if oc[s]!=nc.get(s)]
print(f"- penetapan kasus berubah: **{len(d)}** {'(identik — pin bekerja)' if not d else d}")
print(f"- counts sama: {o['counts']==n['counts']}")
print("- paket P1 + kedua workbook psikolog dibangun ulang")
print()
PY

if [ "$SKIP_RAGAS" = "1" ]; then
  say "SKIP_RAGAS=1 — berhenti setelah langkah 3"
  add "## 4-5. Dilewati (SKIP_RAGAS=1)"; add ""
  add "Selesai $(date '+%Y-%m-%d %H:%M')."; exit 0
fi

# ------------------------------------------------------------- 4. RAGAS ----
say "== 4. RAGAS (judge Qwen) =="
pkill -f "chatbot_app.py" 2>/dev/null && sleep 3
gpu_free || fail "membebaskan GPU sebelum Qwen"
serve "Qwen/Qwen3-14B" || fail "menjalankan Qwen"

cp -f outputs/analysis/ragas_chatbot.csv        "outputs/logs/ragas_prev_${STAMP}.csv" 2>/dev/null
.venv-ragas/bin/python scripts/ragas_eval.py \
  --answers outputs/analysis/study_answers_chatbot.jsonl --tag chatbot --no-correctness \
  >> "$LOG" 2>&1 || fail "RAGAS"

add "## 4. RAGAS"
add ""
sed -n '/## Mean scores/,/^$/p' outputs/analysis/ragas_chatbot_summary.md >> "$REPORT"
add ""
add "Pembanding pra-sesi (judge sama, Qwen3-14B): faithfulness 0.554 · answer_relevancy 0.801 · context_precision 0.727"
add ""

# ------------------------------------------- 5. generator + links kembali ---
say "== 5. mengembalikan Gemma + link konselor =="
gpu_free || fail "membebaskan GPU sebelum Gemma"
bash scripts/serve_chatbot.sh counselor1 >> "$LOG" 2>&1 || fail "link counselor1"
bash scripts/serve_chatbot.sh counselor2 >> "$LOG" 2>&1 || fail "link counselor2"

add "## 5. Link chatbot konselor (baru)"
add ""
add '```'
bash scripts/serve_chatbot.sh link >> "$REPORT" 2>>"$LOG"
add '```'
add ""
add "> Link mati bersama job SLURM ini. Jangan dibagikan jauh sebelum sesi."
add ""

# ------------------------------------------------------------- 6. selesai --
add "---"
add ""
add "Selesai **$(date '+%Y-%m-%d %H:%M')**. Log lengkap: \`${LOG#$ROOT/}\`"
say "SELESAI — laporan: ${REPORT#$ROOT/}"
