#!/bin/bash
# Intervention percentages to try
INTERVENTION_PCTS=(0.25 0.4)

# Models
MODELS=("Qwen/Qwen2.5-0.5B")

# Configs: "stability_weight mi_weight carma_weight label"
CONFIGS=(
  "1.0 0.0 0.0 StabilityOnly"
  "0.0 1.0 0.0 MIONly"
)
DATASET_NAME="idm"
DATA_PATH="./data/wordnet_data_definitions.json"
BATCH_SIZE=64
NUM_EPOCHS=3
SAVE_TEST_RESULTS="--save_test_results"
INTERVENTION="--intervention"
INTERVENTION_TYPE="--intervention_type synonym"
LEARNING_RATE=6e-5
WARMUP_STEPS=500
SAVE_MODEL=False
SAVE_PATH="./models"
MAX_GRAD_NORM=1.0
CALCULATE_LAYERS_PERCENTAGE=False

for SEED in 32 123 111 42 60; do
  for MODEL_NAME in "${MODELS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
      set -- $CONFIG
      STABILITY_WEIGHT=$1
      MI_WEIGHT=$2
      CARMA_WEIGHT=$3
      LABEL=$4

      for INTERVENTION_PCT in "${INTERVENTION_PCTS[@]}"; do
        echo "Running $LABEL | model=$MODEL_NAME | seed=$SEED | intervention_pct=$INTERVENTION_PCT | stability_weight=$STABILITY_WEIGHT | mi_weight=$MI_WEIGHT | carma_weight=$CARMA_WEIGHT"

        CMD="python -m src.main \
          --model_name $MODEL_NAME \
          --dataset_name $DATASET_NAME \
          --data_path $DATA_PATH \
          --batch_size $BATCH_SIZE \
          --train \
          --test \
          --num_epochs $NUM_EPOCHS \
          --learning_rate $LEARNING_RATE \
          --warmup_steps $WARMUP_STEPS \
          --seed $SEED \
          --stability_end_layer 5 \
          --stability_weight $STABILITY_WEIGHT \
          --mi_end_layer 5 \
          --mi_weight $MI_WEIGHT \
          --carma_weight 0.5 \
          --intervention_percentage $INTERVENTION_PCT \
          --save_model \
          $SAVE_TEST_RESULTS \
          $INTERVENTION \
          $INTERVENTION_TYPE \
          --save_test_results \
          --max_grad_norm $MAX_GRAD_NORM \
          --correct_pred_path \"$CORRECT_PRED_PATH\""
        echo "$CMD"
        eval $CMD

      done
    done
  done
done
