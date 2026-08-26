#!/usr/bin/env bash
# Run the depression chatbot and print a fresh public gradio.live link.
#
# Usage:
#   bash scripts/serve_chatbot.sh              # ONE shared link (port 7860)
#   bash scripts/serve_chatbot.sh 7870         # ...on a different port
#   bash scripts/serve_chatbot.sh counselor1   # dedicated link for counselor 1 (port 7861, own logs)
#   bash scripts/serve_chatbot.sh counselor2   # dedicated link for counselor 2 (port 7862, own logs)
#   bash scripts/serve_chatbot.sh link         # show the CURRENT saved link(s)
#   bash scripts/serve_chatbot.sh stop         # stop ALL app instances (leaves vLLM running)
#   bash scripts/serve_chatbot.sh stop counselor1   # stop just that instance
#
# For a study with two counselors online at once, run `counselor1` and
# `counselor2`: they share the one vLLM generator but get separate links,
# separate ports, and SEPARATE interaction logs (outputs/chat_logs/<instance>/)
# so you can tell whose sessions are whose. Each counselor mode does NOT kill
# the other. All processes keep running in the background after this exits.
# Each run makes a NEW URL (gradio.live links last up to ~1 week).

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export HF_HOME="$ROOT/.hf_cache"
export GRADIO_ANALYTICS_ENABLED=False
export PYTHONUNBUFFERED=1

VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-12B-it}"
# Pin the weights. .hf_cache already holds THREE snapshots of this checkpoint and
# their tokenizer_config.json differs, so upstream moved the chat template under
# this project at least once. Unpinned, a new revision on study day means either
# a ~23 GB download that blows the startup wait, or a silently changed template —
# and the live tool is then no longer the instrument that produced the 50 frozen
# briefings the psychologists rate. Verified present in .hf_cache 2026-08-08.
VLLM_REVISION="${VLLM_REVISION:-707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
mkdir -p "$ROOT/outputs/logs"

# Check WHICH model is served, not merely that :8000 answers. A different model
# (e.g. the RAGAS judge) keeps the port alive, and a bare reachability check would
# skip starting the generator — silently backing the counselor chatbot with the
# wrong model.
vllm_up() { curl -sf --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null \
             | grep -qF "\"$VLLM_MODEL\""; }
link_file() { echo "$ROOT/outputs/CURRENT_GRADIO_LINK${1:+_$1}.txt"; }

# ---- resolve the requested instance -> port, label, log dir ------------------
INSTANCE="shared"; PORT=7860; LOGDIR="outputs/chat_logs"
case "${1:-}" in
  counselor1) INSTANCE=counselor1; PORT=7861; LOGDIR="outputs/chat_logs/counselor1" ;;
  counselor2) INSTANCE=counselor2; PORT=7862; LOGDIR="outputs/chat_logs/counselor2" ;;
  link)
    found=0
    # the shared entry ends with ONE colon so its suffix parses to "" — with two
    # it parsed to ":" and this loop looked for CURRENT_GRADIO_LINK_:.txt, so the
    # shared link was never printed
    for pair in "shared:7860:" "counselor1:7861:counselor1" "counselor2:7862:counselor2"; do
      inst="${pair%%:*}"; rest="${pair#*:}"; port="${rest%%:*}"; suffix="${rest#*:}"
      f="$(link_file "$suffix")"
      [ -s "$f" ] || continue; found=1
      pgrep -f "chatbot_app.py --share --port $port" >/dev/null 2>&1 && alive="running" || alive="NOT running"
      printf '%-11s %s  (port %s, %s)\n' "$inst:" "$(cat "$f")" "$port" "$alive"
    done
    [ "$found" = 1 ] || echo "No saved link yet. Generate one with: bash scripts/serve_chatbot.sh"
    exit 0 ;;
  stop)
    case "${2:-all}" in
      counselor1) pkill -f "chatbot_app.py --share --port 7861" && echo "counselor1 stopped." || echo "counselor1 not running." ;;
      counselor2) pkill -f "chatbot_app.py --share --port 7862" && echo "counselor2 stopped." || echo "counselor2 not running." ;;
      all|"")     pkill -f "chatbot_app.py" && echo "all app instances stopped (vLLM left running)." || echo "no app process found." ;;
    esac
    exit 0 ;;
  "")         : ;;                              # default shared instance
  *)          PORT="${1}" ;;                    # numeric port for the shared instance
esac
LINK_FILE="$(link_file "$([ "$INSTANCE" = shared ] && echo "" || echo "$INSTANCE")")"

# ---- 0. login is optional (study decision, 2026-08-08) -----------------------
# Counselors work from the frozen, pre-anonymized question set, and handing out
# per-person credentials cost more setup friction than the gate bought. The
# mechanism is kept: export CHATBOT_AUTH="user:pass" before running and this
# script re-enables the login automatically.
#
# Credentials, when used, still travel by environment and never argv — `ps` and
# /proc/<pid>/cmdline are readable by every other account on this node.
AUTH_FLAG=""
if [ -n "${CHATBOT_AUTH:-}" ]; then
  AUTH_FLAG="--require-auth"
  echo "== login ENABLED for this instance (\$CHATBOT_AUTH is set) =="
else
  echo "== login DISABLED: the public link will be open to anyone who has it. =="
  echo "   Send it only to the study counselors. To require a login instead:"
  echo "     export CHATBOT_AUTH=\"konselor1:<password>\"  &&  re-run this script"
fi

echo "== depression chatbot [$INSTANCE, port $PORT]: generating a fresh public link =="

# ---- 1. ensure the vLLM generator is up (shared by all instances) ------------
if vllm_up; then
  echo "[1/3] vLLM generator already up on :8000"
else
  echo "[1/3] vLLM not running — starting it (first start takes ~4-5 min)…"
  VLOG="$ROOT/outputs/logs/vllm_$(date +%Y%m%d-%H%M%S).log"
  # --host 127.0.0.1: vLLM defaults to 0.0.0.0 and serves no API key, so the
  # default exposes the study generator to every account that can reach this
  # node. Only this machine's app instances need it.
  # VLLM_ATTENTION_BACKEND is NOT set: vLLM 0.24 logs it as an unknown variable
  # and picks the backend itself (TRITON_ATTN for gemma-4's heterogeneous head
  # dims). VLLM_USE_FLASHINFER_SAMPLER=0 is still honoured and still required —
  # this node has no nvcc, so the default flashinfer sampler fails to build.
  VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup .venv-vllm/bin/vllm serve "$VLLM_MODEL" --revision "$VLLM_REVISION" \
      --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 16384 \
      --host 127.0.0.1 --port 8000 > "$VLOG" 2>&1 &
  disown
  echo "      waiting for vLLM (log: $VLOG)…"
  # 90 x 10 s = 15 min, not 10. A cold start was MEASURED at 520 s on an idle
  # A40 (2026-08-08) against the old 600 s ceiling — 80 s of margin, which a
  # busier node or a cold page cache eats easily. The abort below is the only
  # thing standing between a slow start and a published dry-run link, so it must
  # fire on a real failure, not on a slow success.
  for _ in $(seq 1 90); do vllm_up && break; sleep 10; done
  if vllm_up; then echo "      vLLM ready."
  else
    # Do NOT fall through. Without a generator the app still launches, still
    # mints a public link, and answers clinical questions with the raw assembled
    # prompt — and the LLM risk stage is skipped entirely, leaving only the
    # keyword screen. Publishing that to a counselor is worse than not starting.
    echo "      ABORT: vLLM did not come up within 15 minutes." >&2
    echo "      No link was published. Check the log: $VLOG" >&2
    exit 1
  fi
fi

# ---- 2. restart ONLY this instance (matched by its own port) -----------------
echo "[2/3] stopping any previous [$INSTANCE] instance on port $PORT…"
pkill -f "chatbot_app.py --share --port $PORT" 2>/dev/null && sleep 4 || true

# ---- 3. launch with a fresh share link + this instance's own log folder ------
mkdir -p "$ROOT/$LOGDIR"
echo "[3/3] launching [$INSTANCE] with a public share link on port $PORT…"
LOG="$ROOT/outputs/logs/share_${INSTANCE}_$(date +%Y%m%d-%H%M%S).log"
# Embed the query on CPU: the app only encodes one short query at a time, and
# generation runs remotely in vLLM — so the app needs no GPU. This frees GPU
# memory for vLLM and lets several counselor instances run at once (each GPU
# embedder would otherwise take ~2.3 GiB and OOM against vLLM's 85% reservation).
# $AUTH_FLAG goes LAST and may be empty: the stop/link subcommands match
# instances with `pkill -f "chatbot_app.py --share --port $PORT"`, so nothing may
# be inserted before --log-dir. CHATBOT_AUTH is inherited from this shell's
# environment and never appears on the command line.
CUDA_VISIBLE_DEVICES="" nohup .venv-chat/bin/python -u scripts/chatbot_app.py \
  --share --port "$PORT" --log-dir "$LOGDIR" $AUTH_FLAG > "$LOG" 2>&1 &
disown
# The share log records the live gradio.live URL, which is the whole credential
# now that there is no password. outputs/ carries default:group::rwx, so a plain
# redirect leaves it readable by every member of the unix group.
chmod 600 "$LOG" 2>/dev/null; setfacl -b "$LOG" 2>/dev/null || true

URL=""
for _ in $(seq 1 40); do
  # DRY-RUN first, and before the URL check: the app prints this and then still
  # mints a link. A dry-run instance answers with the raw assembled prompt and
  # skips the LLM risk stage for the whole session, leaving only the keyword
  # screen — publishing that to a counselor is the failure this whole script
  # exists to prevent. chatbot_app.py refuses too; this is the belt to that
  # brace, and it kills the process so no live tunnel is left behind.
  if grep -q "continuing in DRY-RUN" "$LOG" 2>/dev/null; then
    pkill -f "chatbot_app.py --share --port $PORT" 2>/dev/null
    echo "ABORT: the app came up in DRY-RUN (no generator reachable)." >&2
    echo "       Instance killed, no link published. See $LOG" >&2
    exit 1
  fi
  URL="$(grep -oE 'https://[a-z0-9]+\.gradio\.live' "$LOG" 2>/dev/null | head -1)"
  [ -n "$URL" ] && break
  # gradio's actual wording is "Could not create share link". The old pattern
  # also matched a bare "Error", which fires on harmless warning lines.
  if grep -qi "could not create share" "$LOG" 2>/dev/null; then
    echo "ERROR: gradio could not create the share link — see $LOG"; exit 1
  fi
  sleep 5
done

echo
if [ -n "$URL" ]; then
  printf '%s\n' "$URL" > "$LINK_FILE"
  # outputs/ carries default:group::rwx, so a plain write here is readable AND
  # writable by every member of the unix group. On a public share link the URL
  # is half the credential, so it gets the same treatment as the password file.
  chmod 600 "$LINK_FILE" 2>/dev/null; setfacl -b "$LINK_FILE" 2>/dev/null || true

  # ---- session record: link + credentials, regenerated every run ------------
  # Written unconditionally rather than kept by hand: a leftover file from an
  # earlier launch advertises a dead URL and a password that opens nothing,
  # which reads as current and is worse than having no file at all.
  CRED_FILE="${LINK_FILE/CURRENT_GRADIO_LINK/SESSION_CREDENTIALS}"
  JOB_END="unknown (not running under SLURM?)"
  if command -v squeue >/dev/null 2>&1; then
    JOB_END="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o '%e' 2>/dev/null \
               || squeue -u "$USER" -h -o '%e' 2>/dev/null | head -1)"
    [ -n "$JOB_END" ] || JOB_END="unknown"
  fi
  # Create empty and lock down BEFORE writing the password, so it is never
  # briefly group-readable between creation and chmod.
  : > "$CRED_FILE"; chmod 600 "$CRED_FILE" 2>/dev/null
  setfacl -b "$CRED_FILE" 2>/dev/null || true
  {
    echo "CHATBOT SESSION — $INSTANCE (port $PORT)"
    echo "==========================================================================="
    echo "CONFIDENTIAL — mode 0600 on purpose. A copy of this file inherits the"
    echo "outputs/ group ACL instead; re-apply 'chmod 600' if you move it."
    echo "outputs/ is gitignored, so this cannot be committed by accident."
    echo "Regenerated on every launch — do not edit, it will be overwritten."
    echo
    echo "LINK       $URL"
    echo "LOGS       $LOGDIR/"
    echo "GENERATED  $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    echo "ACCESS"
    echo "---------------------------------------------------------------------------"
    # ${CHATBOT_AUTH:-} , not $CHATBOT_AUTH : `set -u` is on and login is now
    # optional, so the unset case has to be the normal path, not an error.
    if [ -z "${CHATBOT_AUTH:-}" ]; then
      echo "  NO LOGIN. Anyone holding the URL above can use this instance and"
      echo "  read its answers. The URL is therefore the whole credential:"
      echo "    * send it only to the counselor it is for, and to nobody else;"
      echo "    * do not paste it into shared docs, chats or issue trackers;"
      echo "    * re-run this script when the study ends to retire it."
      echo
      echo "  To require a login instead, relaunch from a shell with:"
      echo "    export CHATBOT_AUTH=\"konselor1:<password>\""
    else
      n=0
      IFS=',' read -ra _pairs <<< "${CHATBOT_AUTH:-}"
      for _p in "${_pairs[@]}"; do
        printf '  %-20s %s\n' "${_p%%:*}" "${_p#*:}"
        n=$((n+1))
      done
      if [ "$n" -gt 1 ]; then
        echo
        echo "  NOTE: \$CHATBOT_AUTH held $n pairs, so THIS instance accepts ALL of"
        echo "  them — credentials are per-process, not per-counselor. If you need a"
        echo "  session to be attributable to one person, launch each instance from a"
        echo "  shell exporting only that person's pair."
      fi
      echo
      echo "  Send the password over a DIFFERENT channel than the link."
    fi
    echo
    echo "EXPIRES — whichever comes first"
    echo "---------------------------------------------------------------------------"
    echo "  1. SLURM job ends      $JOB_END"
    echo "     SLURM kills the whole job cgroup at the walltime limit; nohup does"
    echo "     not survive it. vLLM and the app die together and this URL goes dead."
    echo "  2. Share link expires  ~1 week from GENERATED above (best effort)"
    echo
    echo "  Verify it is still served (a live process does not prove a live tunnel):"
    echo "    curl -s -o /dev/null -w '%{http_code}\\n' -X POST \\"
    echo "      $URL/gradio_api/run/predict \\"
    echo "      -H 'Content-Type: application/json' -d '{\"data\":[\"x\"],\"fn_index\":0}'"
    echo "    # 401 = up and correctly gated.   000 = tunnel dead."
    echo
    echo "RELAUNCHING"
    echo "---------------------------------------------------------------------------"
    echo "  Every relaunch mints a NEW URL that must be re-sent to participants."
    echo "  The app stores no password: it lives only in the running process's"
    echo "  environment, so you choose it again at launch."
    echo
    echo "  For a run longer than one sitting, submit the chatbot as its own job:"
    echo "    export CHATBOT_AUTH=\"konselor1:<pw1>,konselor2:<pw2>\""
    echo "    sbatch --export=ALL scripts/serve_chatbot_slurm.sbatch"
    echo
    echo "WHAT ACCESS CONTROL DOES NOT COVER"
    echo "---------------------------------------------------------------------------"
    echo "  A gradio.live link relays every message and every generated briefing"
    echo "  through a third-party server, so text typed here leaves the cluster."
    echo "  A login would control WHO can use the tool, never where the text goes,"
    echo "  so removing it does not change this. The safeguard that does is the"
    echo "  protocol: counselors work from the frozen, pre-anonymized question set"
    echo "  and are told in the UI not to type real client data."
    echo "  Name gradio.live as a processor in the ethics materials and the"
    echo "  participant briefing, or drop --share and use the SSH tunnel"
    echo "  (documents/running_the_chatbot.md section 4)."
  } >> "$CRED_FILE"

  echo "=================================================================="
  echo "  INSTANCE:     $INSTANCE   (port $PORT)"
  echo "  PUBLIC LINK:  $URL"
  echo "  saved to:     $LINK_FILE"
  echo "  credentials:  $CRED_FILE  (mode 0600)"
  echo "  logs in:      $LOGDIR/"
  echo "  (temporary — lasts up to ~1 week; re-run this script for a new one)"
  echo "=================================================================="
else
  echo "No URL appeared after waiting. Check the log: $LOG"; exit 1
fi
