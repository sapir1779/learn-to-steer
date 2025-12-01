#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "Starting at $TIMESTAMP"


# Define model-experiment pairs
declare -a preprocessing_folders=(
    "/cortex/users/sapiry7/workspace/GQA_preprocessing/20250507_123136__IS_A__WITH_IS_RELATION__2000_400_200_DATA/"
    # "/cortex/users/sapiry7/workspace/GQA_preprocessing/20250510_193824__IS_A__WITH_IS_RELATION__800_200_100_DATA/"
    "/cortex/users/sapiry7/workspace/ORM_preprocessing/20250507_122709__above_below_left_right__white_green__800_100_70_data/"
)

declare -a experiment_names=(
    "REFACTORED__GQA__IS_A__2000_400_200"
    # "REFACTORED__GQA__IS_A__800_200_100"
    "REFACTORED__ORM__above_below_left_right__white_green__800_100_70"
)

declare -a train_sizes=(
    "10000"
    "4000"
    # "4000"
)

declare -a val_sizes=(
    "2000"
    "500"
    # "500"
)

declare -a test_sizes=(
    "1000"
    "350"
    # "350"
)
# Check if arrays have the same length
if [ ${#preprocessing_folders[@]} -ne ${#experiment_names[@]} ]; then
    echo "Error: preprocessing_folders and experiment_names must have the same length"
    exit 1
fi


SCRIPT_PATH="train_relation_classifier/data_creation/attention_maps_generation/generate_relations_data.py"

OUTPUT_MAP_RESOLUTION=-1
# OUTPUT_MAP_RESOLUTION=16

TARGET_TIMESTEPS="1 3"
# TARGET_TIMESTEPS="5 10 15 20 25 49"
TARGET_TIMESTEPS_ARGS="${TARGET_TIMESTEPS}"
target_timesteps_str="${TARGET_TIMESTEPS_ARGS// /_}"

MAP_SIZE="16"
DIFFUSION_VERSION="SCHNELL"
# MAP_SIZE="32"
# DIFFUSION_VERSION="DEV"
# MAP_SIZE="16"
# DIFFUSION_VERSION="SD2"

# Loop over preprocessing folder <--> experiment pairs
for i in "${!preprocessing_folders[@]}"; do
    preprocessing_folder=${preprocessing_folders[$i]}
    train_size=${train_sizes[$i]}
    val_size=${val_sizes[$i]}
    test_size=${test_sizes[$i]}
    TRAIN_PATH="${preprocessing_folder}/train__size_${train_size}/result_metadata.json"
    VAL_PATH="${preprocessing_folder}/val__size_${val_size}/result_metadata.json"
    TEST_PATH="${preprocessing_folder}/test__size_${test_size}/result_metadata.json"

    GENERAL_EXP_NAME="${experiment_names[$i]}__T_${target_timesteps_str}"
    if [ "$MAP_SIZE" = "32" ]; then
        GENERAL_EXP_NAME="${MAP_SIZE}__${GENERAL_EXP_NAME}"
    fi
    OUTPUT_PATH="/cortex/users/sapiry7/workspace/RELATION_CLASSIFIER_TRAINING_DATA/${DIFFUSION_VERSION}__${GENERAL_EXP_NAME}"

    EXP_NAME_TRAIN="${TIMESTAMP}__train__${train_size}"
    EXP_NAME_VAL="${TIMESTAMP}__val__${val_size}"
    EXP_NAME_TEST="${TIMESTAMP}__test__${test_size}"

    ARGS=""
    ARGS="${ARGS} --output_path ${OUTPUT_PATH} "
    ARGS="${ARGS} --diffusion_version ${DIFFUSION_VERSION} "
    ARGS="${ARGS} --map_size ${MAP_SIZE} "
    ARGS="${ARGS} --output_map_resolution ${OUTPUT_MAP_RESOLUTION} "
    ARGS="${ARGS} --target_timesteps ${TARGET_TIMESTEPS_ARGS} "

    echo "Creating train/val/test data"

    python ${SCRIPT_PATH} \
        ${ARGS} \
        --metadata_path ${TRAIN_PATH} \
        --gpu_indices 0 1 2 3 4 5 6 7 \
        --experiment_name ${EXP_NAME_TRAIN} ; \
    python ${SCRIPT_PATH} \
        ${ARGS} \
        --metadata_path ${VAL_PATH} \
        --gpu_indices 0 1 2 3 4 \
        --experiment_name ${EXP_NAME_VAL} & \
    python ${SCRIPT_PATH} \
        ${ARGS} \
        --metadata_path ${TEST_PATH} \
        --gpu_indices 5 6 7 \
        --experiment_name ${EXP_NAME_TEST} & \
    wait

done
