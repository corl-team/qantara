#!/usr/bin/env bash
# Timing driver: per-dispatch wall-clock across the 4 envs (headline checkpoints).
# Runs eval.py with a small budget (timing is deterministic in expectation; SR is
# not the deliverable here) and headline solver settings. eval.py's timing wrapper
# (eval.py:20-39,247-267) writes a sidecar JSON per (env, dispatch).
#
# Output sidecars: $STABLEWM_HOME/<policy_dir>/<env>_results.txt.seed<S>.timing.json
# Aggregator: scripts/agg_timing.py reads sidecars across (env, dispatch).
set -euo pipefail

# Per-env ckpt: each env-specific model has its own action_dim. POLICY_DIR_TPL
# is a template with literal ${env} placeholder (escape via \${env} so bash keeps it).
POLICY_DIR_TPL=${POLICY_DIR_TPL:-"qantara-\${env}-nulldrop-0p0-seed11"}
SEEDS=${SEEDS:-"42 2026 3407"}                                # eval seeds (each emits one sidecar)
NUM_EVAL=${NUM_EVAL:-5}                                       # tiny — timing not SR
ROLLOUT_K=${ROLLOUT_K:-4}                                     # CEM x̂-recursion (headline)
ROLLOUT_A_K=${ROLLOUT_A_K:-2}                                 # BC Euler steps (headline)
ENVS=(${ENVS:-pusht tworoom cube reacher})

# Per-env dispatch loop. Each call writes to STABLEWM_HOME/<POLICY_DIR>/<env>_results.txt
# (existing convention) plus the new .seed<S>.timing.json sidecar. SUBDIR per dispatch
# keeps the BC and video_idm sidecars from clobbering CEM's.
run_one() {
    local env="$1" pk="$2" bk="$3" seed="$4"
    # Resolve env-specific policy dir from the template.
    local policy_dir="${POLICY_DIR_TPL//\$\{env\}/$env}"
    local sub="${policy_dir}_${5}"
    local ckpt_src="${STABLEWM_HOME:?}/${policy_dir}/qantara_object.ckpt"
    [ -e "$ckpt_src" ] || { echo "[skip] missing $ckpt_src"; return 0; }
    mkdir -p "${STABLEWM_HOME}/${sub}"
    [ -e "${STABLEWM_HOME}/${sub}/qantara_object.ckpt" ] || ln -s "$ckpt_src" "${STABLEWM_HOME}/${sub}/qantara_object.ckpt"

    local extra=()
    if [ "$pk" = "bc" ]; then
        extra+=("+policy_kind=bc" "+bc_kind=${bk}")
    fi
    echo "=== ${env} / dispatch=${pk}${bk:+/$bk} / seed=${seed} ==="
    uv run python eval.py --config-name="$env" \
        policy="${sub}/qantara" seed="$seed" \
        eval.num_eval="$NUM_EVAL" \
        +eval.save_video=false \
        +wm.rollout_k="$ROLLOUT_K" +wm.rollout_a_k="$ROLLOUT_A_K" \
        "${extra[@]}" \
        hydra.output_subdir=null hydra.run.dir=. 2>&1 | tail -3
}

for env in "${ENVS[@]}"; do
    for seed in $SEEDS; do
        run_one "$env" cem "" "$seed" cem
        run_one "$env" bc bc "$seed" bc
        run_one "$env" bc video_idm "$seed" vidim
    done
done

echo
echo "Sidecar JSONs:"
find "${STABLEWM_HOME}" -maxdepth 2 -name "*timing.json" 2>/dev/null | head -20
