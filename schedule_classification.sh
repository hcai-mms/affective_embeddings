#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [train|test] [macro|micro] [elasticnet|knn|mlp|xgboost]"
    echo ""
    echo "Example:"
    echo "  $0 train macro mlp    # Schedule MLP training jobs for GoEmotions"
    exit 1
fi

MODE=$1
CATEGORY=$2
MODEL=$3

# Validate arguments
if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "ERROR: First argument must be 'train' or 'test'"
    exit 1
fi

if [[ "$CATEGORY" != "macro" && "$CATEGORY" != "micro" ]]; then
    echo "ERROR: Second argument must be 'macro' or 'micro'"
    exit 1
fi

if [[ "$MODEL" != "elasticnet" && "$MODEL" != "knn" && "$MODEL" != "mlp" && "$MODEL" != "xgboost" ]]; then
    echo "ERROR: Third argument must be 'elasticnet', 'knn', 'mlp', or 'xgboost'"
    exit 1
fi

EMBEDDINGS_DIR="embeddings"
SPLITS_PATH="opendata/splits/stratified_splits_5fold_go_emotion.parquet"

if [[ "$MODEL" == "elasticnet" ]]; then
    MODEL_JOBS=-1
    CV_JOBS=5
    OPTUNA_JOBS=1
    N_TRIALS=100
    TIMEOUT=600
    SLURM_SCRIPT="run_slurm.sh"
elif [[ "$MODEL" == "knn" ]]; then
    MODEL_JOBS=4
    CV_JOBS=5
    OPTUNA_JOBS=1
    N_TRIALS=100
    TIMEOUT=300
    SLURM_SCRIPT="run_slurm_non_exclusive.sh"
elif [[ "$MODEL" == "mlp" ]]; then
    MODEL_JOBS=1
    CV_JOBS=5
    OPTUNA_JOBS=5
    N_TRIALS=200
    TIMEOUT=600
    SLURM_SCRIPT="run_slurm_non_exclusive.sh"
elif [[ "$MODEL" == "xgboost" ]]; then
    MODEL_JOBS=1
    CV_JOBS=1
    OPTUNA_JOBS=1
    N_TRIALS=100
    TIMEOUT=600
    SLURM_SCRIPT="run_slurm_gpu.sh"
else
    MODEL_JOBS=1
    CV_JOBS=1
    OPTUNA_JOBS=1
    N_TRIALS=100
    TIMEOUT=60
    SLURM_SCRIPT="run_slurm_non_exclusive.sh"
fi

# Find all embedding files
EMBEDDINGS=($EMBEDDINGS_DIR/embeddings_*.parquet)

echo "====================================================================="
echo "Scheduling ${CATEGORY^^} Experiments - ${MODE^^} Mode"
echo "====================================================================="
echo "Model: $MODEL"
echo "Embeddings: ${#EMBEDDINGS[@]} files"
echo "Splits: ${#SPLITS[@]} types"
if [ "$MODE" == "train" ]; then
    echo "Mode: Training (folds 0-4)"
    echo "N_TRIALS: $N_TRIALS"
    echo "MODEL_JOBS: $MODEL_JOBS (model training cores)"
    echo "CV_JOBS: $CV_JOBS (model training cores)"
    echo "OPTUNA_JOBS: $OPTUNA_JOBS (parallel trials)"
else
    echo "Mode: Test evaluation"
fi
echo "====================================================================="
echo ""

TOTAL_JOBS=0

for embedding_path in "${EMBEDDINGS[@]}"; do
    embedding_name=$(basename "$embedding_path")
    echo "Embedding: $embedding_name"
    if [ "$MODE" == "train" ]; then
        sbatch -J "emo_${CATEGORY}_${MODEL}" $SLURM_SCRIPT services.classification_benchmark_service \
            --embedding-path "$embedding_path" \
            --splits-path "$SPLITS_PATH" \
            --category "$CATEGORY" \
            --model "$MODEL" \
            --train \
            --n-trials "$N_TRIALS" \
            --model-jobs "$MODEL_JOBS" \
            --cv-jobs "$CV_JOBS" \
            --optuna-jobs "$OPTUNA_JOBS" \
            --timeout "$TIMEOUT"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    else
        # Schedule test evaluation job
        echo "    Scheduling test evaluation (fold -1)..."

        sbatch $SLURM_SCRIPT services.classification_benchmark_service \
            --embedding-path "$embedding_path" \
            --splits-path "$SPLITS_PATH" \
            --category "$CATEGORY" \
            --model "$MODEL" \
            --test \
            --n-trials "$N_TRIALS" \
            --model-jobs "$MODEL_JOBS" \
            --cv-jobs "$CV_JOBS" \

        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    fi
done

echo "====================================================================="
echo "Scheduled $TOTAL_JOBS ${MODE} jobs for ${CATEGORY^^} ($MODEL)"
echo "====================================================================="
echo ""

if [ "$MODE" == "train" ]; then
    echo "After all training jobs complete, run test evaluation with:"
    echo "  $0 test $CATEGORY $MODEL"
else
    echo "Test evaluation jobs scheduled. Monitor with: squeue -u \$USER"
fi
echo ""
