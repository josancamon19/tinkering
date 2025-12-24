# Ablation study on LoRA rank parameter
RANKS=(
    4
    8
    16
    32
    64
)

# Use the same source as the rankings experiment
SOURCE="openthoughts_all_domains_ai2_adapt_dev_openmath_2_math_t4096_n500"

# Function to run training for a single rank
run_training() {
    local RANK=$1
    echo "----------------------------------------------------------------"
    echo "Running training for lora_rank: $RANK"
    echo "----------------------------------------------------------------"
    
    # Run training with the specified LoRA rank
    uv run python src/tinkering/tinker_openthoughts/train.py \
        dataset_name="$SOURCE" \
        lora_rank="$RANK"
}

# Run trainings in parallel
run_training "${RANKS[0]}" &
run_training "${RANKS[1]}" &
run_training "${RANKS[2]}" &
run_training "${RANKS[3]}" &
run_training "${RANKS[4]}" &

wait

