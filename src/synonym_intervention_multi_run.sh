#!/bin/bash

# List of models, weights, and correct predictions
MODELS=("GPT2-large")
WEIGHTS=(
 ""
)
CORRECT_PREDS=(
 "results/GPT2_large/idm/test_results/GPT2_large_idm_test_results.json"
)

SEEDS=(42 123 19 7 14 23)

# Common flags
DATASET_NAME="idm"
DATA_PATH="./data/wordnet_data_definitions.json"
BATCH_SIZE=16
NUM_EPOCHS=3
INTERVENTION_TYPE="--intervention_type synonym"

# Loop through each model
for i in "${!MODELS[@]}"; do
  MODEL_NAME=${MODELS[$i]}
  MODEL_PATH=${WEIGHTS[$i]}
  CORRECT_PRED_PATH=${CORRECT_PREDS[$i]}
  for seed in "${SEEDS[@]}"; do
    # Run the model
    CMD="python -m src.main \
      --model_name $MODEL_NAME \
      --dataset_name $DATASET_NAME \
      --data_path $DATA_PATH \
      --batch_size $BATCH_SIZE \
      --seed $seed \
      --save_test_results \
      --intervention \
      --intervention_percentage 0.4 \
      $INTERVENTION_TYPE \
      --correct_pred_path $CORRECT_PRED_PATH"\

      # Add model_path only if it is not empty
      if [ -n "$MODEL_PATH" ]; then
          CMD="$CMD --model_path \"$MODEL_PATH\""
      fi
      echo "Evaluating  $CMD"
#      eval $CMD
  done
done
