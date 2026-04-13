#!/bin/bash
#SBATCH -J "cifar_scale"
#SBATCH -o slurm/logs/data_scaling_%A_%a.out
#SBATCH -e slurm/logs/data_scaling_%A_%a.err
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-44

# Data-scaling sweep for CIFAR-10 architectural evolution.
#
# Grid: {resnet, swin, hybrid} x {1K, 5K, 10K, 25K, 50K} x {3 seeds} = 45 runs
# Each array task trains one (model, subset_size, seed) triple independently.
#
# Auto-resumes: SLURM sends USR1 5 min before time limit,
# the trap saves checkpoint and requeues this specific array task.
#
# Prerequisites (run once on login node):
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user pytorch-lightning torchmetrics fvcore rich
#
# Submit:  sbatch slurm/data_scaling.sl
# Monitor: tail -f slurm/logs/data_scaling_<jobid>_<taskid>.out

MODELS=(resnet swin hybrid)
SIZES=(1000 5000 10000 25000 50000)
SEEDS=(0 1 2)

NUM_SIZES=${#SIZES[@]}
NUM_SEEDS=${#SEEDS[@]}
PER_MODEL=$(( NUM_SIZES * NUM_SEEDS ))

TASK=$SLURM_ARRAY_TASK_ID
MODEL_IDX=$(( TASK / PER_MODEL ))
REMAIN=$(( TASK % PER_MODEL ))
SIZE_IDX=$(( REMAIN / NUM_SEEDS ))
SEED_IDX=$(( REMAIN % NUM_SEEDS ))

MODEL=${MODELS[$MODEL_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

RUN_NAME="${MODEL}_n${SIZE}_s${SEED}"
CKPT_DIR="checkpoints/${RUN_NAME}"
RESUME_FLAG="${CKPT_DIR}/.resume"

# Auto-resume: the trap creates a sentinel file before requeuing.
# On next run, if the sentinel exists, we pass --ckpt last.
handle_timeout() {
    echo "=== USR1 received ($(date)), saving and requeuing task $TASK ==="
    kill -TERM "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID"
    mkdir -p "$CKPT_DIR"
    touch "$RESUME_FLAG"
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
}
trap handle_timeout USR1

module purge
module load aidl/pytorch/2.6.0-cuda12.6
export PATH="$HOME/.local/bin:$PATH"
pip install --user pytorch-lightning torchmetrics fvcore rich 2>/dev/null

RESUME_ARG=""
if [ -f "$RESUME_FLAG" ]; then
    RESUME_ARG="--ckpt ${CKPT_DIR}/last.ckpt"
    rm "$RESUME_FLAG"
    echo "=== Resuming $RUN_NAME from last.ckpt ==="
fi

echo "=== Task $TASK: model=$MODEL  N=$SIZE  seed=$SEED  ($(date)) ==="
python -u train.py \
    --model "$MODEL" \
    --train-subset "$SIZE" \
    --seed "$SEED" \
    --max-epochs 300 \
    --patience 30 \
    $RESUME_ARG &

TRAIN_PID=$!
wait "$TRAIN_PID"
