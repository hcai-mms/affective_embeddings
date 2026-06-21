#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [train|test|cancel] [plutchik|vad] [elasticnet|knn|mlp|xgboost]"
    echo ""
    echo "Examples:"
    echo "  $0 train plutchik elasticnet    # Schedule ElasticNet training jobs for Plutchik"
    echo "  $0 test plutchik elasticnet     # Schedule ElasticNet test evaluation for Plutchik"
    echo "  $0 train vad mlp            # Schedule GPU-accelerated MLP training jobs for VAD"
    echo "  $0 train plutchik xgboost       # Schedule XGBoost training jobs for Plutchik"
    echo "  $0 cancel                       # Cancel all benchmark_emotion jobs"
    exit 1
fi

MODE=$1
DATASET=$2
MODEL=$3

# Validate arguments
if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "ERROR: First argument must be 'train' or 'test'"
    exit 1
fi

if [[ "$DATASET" != "plutchik" && "$DATASET" != "vad" ]]; then
    echo "ERROR: Second argument must be 'plutchik' or 'vad'"
    exit 1
fi

if [[ "$MODEL" != "elasticnet" && "$MODEL" != "knn" && "$MODEL" != "mlp" && "$MODEL" != "xgboost" ]]; then
    echo "ERROR: Third argument must be 'elasticnet', 'knn', 'mlp', or 'xgboost'"
    exit 1
fi

EMBEDDINGS_DIR="embeddings"
SPLITS_DIR="opendata/splits"

if [[ "$MODEL" == "elasticnet" ]]; then
    MODEL_JOBS=1
    CV_JOBS=5
    OPTUNA_JOBS=2
    N_TRIALS=100
    TIMEOUT=1200
    SLURM_SCRIPT="run_slurm_non_exclusive.sh"
elif [[ "$MODEL" == "knn" ]]; then
    MODEL_JOBS=4
    CV_JOBS=5
    OPTUNA_JOBS=1
    N_TRIALS=100
    TIMEOUT=600
    SLURM_SCRIPT="run_slurm_non_exclusive.sh"
elif [[ "$MODEL" == "mlp" ]]; then
    MODEL_JOBS=1
    CV_JOBS=5
    OPTUNA_JOBS=1
    N_TRIALS=200
    TIMEOUT=1200
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
    TIMEOUT=90
    SLURM_SCRIPT="run_slurm_non_exclusive.sh"
fi

PLUTCHIK_SPLITS=(
    "$SPLITS_DIR/cluster_splits_5fold_community_plutchik.parquet"
    "$SPLITS_DIR/cluster_splits_5fold_morphological_plutchik.parquet"
)

VAD_SPLITS=(
    "$SPLITS_DIR/cluster_splits_5fold_community_vad.parquet"
    "$SPLITS_DIR/cluster_splits_5fold_morphological_vad.parquet"
)

# Select splits based on dataset
if [ "$DATASET" == "plutchik" ]; then
    SPLITS=("${PLUTCHIK_SPLITS[@]}")
else
    SPLITS=("${VAD_SPLITS[@]}")
fi

# Find all embedding files
EMBEDDINGS=($EMBEDDINGS_DIR/embeddings_*.parquet)

echo "====================================================================="
echo "Scheduling ${DATASET^^} Experiments - ${MODE^^} Mode"
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

    for splits_path in "${SPLITS[@]}"; do
        splits_name=$(basename "$splits_path")
        echo "  Split: $splits_name"

        if [ "$MODE" == "train" ]; then
            sbatch -J "emo_${DATASET}_${MODEL}" $SLURM_SCRIPT services.regression_benchmark_service \
                --embedding-path "$embedding_path" \
                --splits-path "$splits_path" \
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

            sbatch $SLURM_SCRIPT services.regression_benchmark_service \
                --embedding-path "$embedding_path" \
                --splits-path "$splits_path" \
                --model "$MODEL" \
                --test \
                --n-trials "$N_TRIALS" \
                --model-jobs "$MODEL_JOBS" \
                --cv-jobs "$CV_JOBS" \

            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        fi
        echo ""
    done
done

echo "====================================================================="
echo "Scheduled $TOTAL_JOBS ${MODE} jobs for ${DATASET^^} ($MODEL)"
echo "====================================================================="
echo ""

if [ "$MODE" == "train" ]; then
    echo "After all training jobs complete, run test evaluation with:"
    echo "  $0 test $DATASET $MODEL"
else
    echo "Test evaluation jobs scheduled. Monitor with: squeue -u \$USER"
fi
echo ""
