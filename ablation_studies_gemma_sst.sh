#!/bin/bash
set -euo pipefail

# Intervention percentages
INTERVENTION_PCTS=(0.25)

# Models
MODELS=("google/gemma-2b")

# Configs: "stability_weight mi_weight carma_weight label"
CONFIGS=(
  "1.0 0.0 0.0 StabilityOnly"
  "0.0 1.0 0.0 MIONly"
)

DATASET_NAME="sst"
DATA_PATH="stanfordnlp/sst"
BATCH_SIZE=16
NUM_EPOCHS=2
LEARNING_RATE=6e-5
WARMUP_STEPS=500
MAX_GRAD_NORM=1.0

SAVE_TEST_RESULTS="--save_test_results"
INTERVENTION="--intervention"
INTERVENTION_TYPE="--intervention_type cap"

CAP_START_LAYERS=(1 4 8 13 17)
GROUPING_PROTOCOLS=(sum mean max)

for SEED in 32 123 42 60; do
  for MODEL_NAME in "${MODELS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
      set -- $CONFIG
      STABILITY_WEIGHT=$1
      MI_WEIGHT=$2
      CARMA_WEIGHT=0.3
      LABEL=$4

      # sanitize decimals for filenames
      SAFE_STABILITY_WEIGHT=${STABILITY_WEIGHT//./_}
      SAFE_MI_WEIGHT=${MI_WEIGHT//./_}
      SAFE_CARMA_WEIGHT=${CARMA_WEIGHT//./_}

      for INTERVENTION_PCT in "${INTERVENTION_PCTS[@]}"; do

        MODEL_PATH="./models/gemma_2b/${DATASET_NAME}/gemma_2b_tw_carma_tuned_${SAFE_CARMA_WEIGHT}_st_${SAFE_STABILITY_WEIGHT}_mi_${SAFE_MI_WEIGHT}_seed_${SEED}.pt"

        CORRECT_PRED_PATH="./results/gemma_2b_tw_carma_tuned_${SAFE_CARMA_WEIGHT}_st_${SAFE_STABILITY_WEIGHT}_mi_${SAFE_MI_WEIGHT}_seed_${SEED}/${DATASET_NAME}/test_results/gemma_2b_tw_carma_tuned_${SAFE_CARMA_WEIGHT}_st_${SAFE_STABILITY_WEIGHT}_mi_${SAFE_MI_WEIGHT}_seed_${SEED}_${DATASET_NAME}_test_results.json"

        CMD="python -m src.main \
          --model_name ${MODEL_NAME} \
          --dataset_name ${DATASET_NAME} \
          --data_path ${DATA_PATH} \
          --batch_size ${BATCH_SIZE} \
          --num_epochs ${NUM_EPOCHS} \
          --learning_rate ${LEARNING_RATE} \
          --warmup_steps ${WARMUP_STEPS} \
          --seed ${SEED} \
          --stability_end_layer 7 \
          --stability_weight ${STABILITY_WEIGHT} \
          --mi_end_layer 7 \
          --mi_weight ${MI_WEIGHT} \
          --carma_weight ${CARMA_WEIGHT} \
          --intervention_percentage ${INTERVENTION_PCT} \
          --max_grad_norm ${MAX_GRAD_NORM} \
          --model_path ${MODEL_PATH} \
          --CAP \
          ${SAVE_TEST_RESULTS} \
          ${INTERVENTION} \
          ${INTERVENTION_TYPE} \
          --CAP_start_layer ${CAP_START_LAYERS[*]} \
          --grouping_protocol ${GROUPING_PROTOCOLS[*]} \
          --correct_pred_path ${CORRECT_PRED_PATH}"

        echo
        echo "=========================================================="
        echo "LABEL=$LABEL"
        echo "MODEL_PATH=$MODEL_PATH"
        echo "CORRECT_PRED_PATH=$CORRECT_PRED_PATH"
        echo "----------------------------------------------------------"
        echo ">>> Running:"
        echo "$CMD"
        echo "=========================================================="

        eval $CMD
      done
    done
  done
done
