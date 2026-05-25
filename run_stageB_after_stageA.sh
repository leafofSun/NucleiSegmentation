#!/usr/bin/env bash
set -euo pipefail

# =========================
# 配置
# =========================
STAGE_A_RUN_NAME="stageA_vision_ddp"
STAGE_B_RUN_NAME="stageB_pnurl_warmup_ddp"

DATA_PATH="data/PanNuke"
KNOWLEDGE_PATH="data/PanNuke/medical_knowledge.json"

GPUS="0,1"
NPROC=2
BATCH_SIZE=2
STAGE_B_EPOCHS=30

# 你已有的 error.log
ERROR_LOG="error.log"

STAGE_A_MODEL_DIR="workdir/models/${STAGE_A_RUN_NAME}"
STAGE_A_BEST="${STAGE_A_MODEL_DIR}/best_model.pth"
STAGE_A_LATEST="${STAGE_A_MODEL_DIR}/latest_model.pth"

GPU_UTIL_THRESHOLD=10
GPU_MEM_THRESHOLD=3000
IDLE_CHECK_INTERVAL=60
IDLE_CONFIRM_TIMES=3

ERROR_PATTERN="Traceback|Error|Exception|Fatal|failed|Failed|CUDA|NCCL|OOM|out of memory|RuntimeError|ValueError|TypeError|KeyError|ImportError|AssertionError|No such file|Killed|Aborted"

log() {
  echo "[$(date '+%F %T')] $*"
}

stage_a_running() {
  pgrep -af "torchrun|train.py" | grep -F -- "--run_name ${STAGE_A_RUN_NAME}" >/dev/null 2>&1
}

choose_stage_a_ckpt() {
  if [[ -f "${STAGE_A_BEST}" ]]; then
    echo "${STAGE_A_BEST}"
    return 0
  fi

  if [[ -f "${STAGE_A_LATEST}" ]]; then
    echo "${STAGE_A_LATEST}"
    return 0
  fi

  return 1
}

gpu_is_idle_once() {
  local busy_count
  busy_count=$(
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v util_th="${GPU_UTIL_THRESHOLD}" -v mem_th="${GPU_MEM_THRESHOLD}" '
      {
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        if ($1 > util_th || $2 > mem_th) busy += 1;
      }
      END { print busy + 0 }
    '
  )

  [[ "${busy_count}" -eq 0 ]]
}

wait_for_stage_a_done() {
  log "等待 Stage A 结束：run_name=${STAGE_A_RUN_NAME}"

  while stage_a_running; do
    log "Stage A 仍在运行，继续等待..."
    sleep 60
  done

  log "未检测到 Stage A 进程，继续检查 checkpoint。"
}

wait_for_checkpoint() {
  log "等待 Stage A checkpoint 生成：${STAGE_A_MODEL_DIR}"

  while true; do
    if CKPT_PATH="$(choose_stage_a_ckpt)"; then
      log "找到 Stage A checkpoint：${CKPT_PATH}"
      echo "${CKPT_PATH}"
      return 0
    fi

    log "还没有找到 best_model.pth 或 latest_model.pth，继续等待..."
    sleep 60
  done
}

wait_for_gpu_idle() {
  log "等待 GPU 空闲：util<=${GPU_UTIL_THRESHOLD}%, mem<=${GPU_MEM_THRESHOLD}MiB，连续 ${IDLE_CONFIRM_TIMES} 次"

  local ok_count=0

  while true; do
    if gpu_is_idle_once; then
      ok_count=$((ok_count + 1))
      log "GPU 空闲检查通过 ${ok_count}/${IDLE_CONFIRM_TIMES}"
    else
      ok_count=0
      log "GPU 仍忙，等待下一次检查..."
    fi

    if [[ "${ok_count}" -ge "${IDLE_CONFIRM_TIMES}" ]]; then
      log "GPU 已稳定空闲，准备启动 Stage B。"
      return 0
    fi

    sleep "${IDLE_CHECK_INTERVAL}"
  done
}

run_stage_b() {
  local resume_ckpt="$1"

  log "启动 Stage B：pnurl_warmup"
  log "Resume: ${resume_ckpt}"
  log "错误写入：${ERROR_LOG}"
  log "训练进度仍显示在当前终端。"

  CUDA_VISIBLE_DEVICES="${GPUS}" \
  OMP_NUM_THREADS=1 \
  PYTHONFAULTHANDLER=1 \
  torchrun --standalone --nproc_per_node="${NPROC}" train.py \
    --phase pnurl_warmup \
    --prompt_mode dynamic_gt \
    --use_pnurl \
    --epochs "${STAGE_B_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --data_path "${DATA_PATH}" \
    --knowledge_path "${KNOWLEDGE_PATH}" \
    --resume "${resume_ckpt}" \
    --run_name "${STAGE_B_RUN_NAME}" \
    2> >(tee >(grep -Eai "${ERROR_PATTERN}" >> "${ERROR_LOG}" || true) >&2)
}

main() {
  wait_for_stage_a_done
  RESUME_CKPT="$(wait_for_checkpoint)"
  wait_for_gpu_idle
  run_stage_b "${RESUME_CKPT}"
}

main "$@"
