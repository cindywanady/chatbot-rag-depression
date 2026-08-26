#!/bin/bash
# Run the counselor chatbot inside its OWN long-lived SLURM job.
#
# WHY THIS EXISTS
# ---------------
# `bash scripts/serve_chatbot.sh` backgrounds vLLM and the app instances with
# nohup, which survives a closed terminal but NOT the end of the SLURM
# allocation they were launched from: SLURM kills the whole job cgroup. Started
# from a VS Code Server desktop job (default walltime 24 h), the chatbot dies
# with it — mid-study, and every restart mints a NEW gradio.live URL that has to
# be re-sent to participants.
#
# This job requests the same A40 node type with the maximum 7-day walltime, so
# one submission covers a whole data-collection window.
#
# USAGE
# -----
#   export CHATBOT_AUTH="konselor1:<pw1>,konselor2:<pw2>"   # NOT in a file
#   sbatch --export=ALL scripts/serve_chatbot_slurm.sbatch
#
#   # once RUNNING (first vLLM start takes ~4-5 min):
#   bash scripts/serve_chatbot.sh link
#
# `--export=ALL` is what carries CHATBOT_AUTH into the job. Without it the job
# aborts rather than publishing an unauthenticated link.
#
# Resources mirror the desktop job this was developed on (Account=lf93,
# 1x A40, 13 CPUs, 240 GB). A40 nodes exist in BOTH the `desktop` and `gpu`
# partitions; `gpu` is used here so a long-running service does not occupy an
# interactive-desktop slot. Adjust the partition/QOS if your allocation differs.

#SBATCH --job-name=chatbot-dep
#SBATCH --account=lf93
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=13
#SBATCH --mem=240G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/logs/slurm_chatbot_%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Login is optional (2026-08-08). If you DO set CHATBOT_AUTH, --export=ALL is
# what carries it into the job; warn rather than abort when it is absent, so the
# no-login path is a deliberate default and a dropped variable is still visible.
if [ -z "${CHATBOT_AUTH:-}" ]; then
  echo "NOTE: CHATBOT_AUTH is not set — both links will be published WITHOUT a login."
  echo "      If you meant to require one, cancel and resubmit with:"
  echo "        export CHATBOT_AUTH=\"konselor1:<pw1>,konselor2:<pw2>\""
  echo "        sbatch --export=ALL scripts/serve_chatbot_slurm.sbatch"
fi

echo "== chatbot job ${SLURM_JOB_ID:-?} on $(hostname), started $(date) =="
echo "   walltime 7 days; links land in outputs/CURRENT_GRADIO_LINK_*.txt"

# serve_chatbot.sh starts vLLM if needed and aborts if it fails to come up, so a
# failed generator fails this job instead of publishing a dry-run link.
bash scripts/serve_chatbot.sh counselor1 || exit 1
bash scripts/serve_chatbot.sh counselor2 || exit 1

echo
echo "== both instances up. Current links: =="
bash scripts/serve_chatbot.sh link

# Hold the allocation open. The app and vLLM processes are children of this job;
# when this script exits, SLURM tears them down. Re-checking on an interval also
# leaves a heartbeat in the job log showing the services were still alive.
while true; do
  sleep 900
  if ! curl -sf --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] WARNING: vLLM is not answering on :8000"
  fi
  for port in 7861 7862; do
    pgrep -f "chatbot_app.py --share --port $port" >/dev/null 2>&1 \
      || echo "[$(date +%H:%M:%S)] WARNING: app instance on port $port is gone"
  done
done
