# Explicitly list the curriculum methods to process
METHODS=(
    "easy_to_hard"
    "first_epoch_only"
    "grouped_shuffle"
)

# Use the same source as the rankings experiment
SOURCE="openthoughts_all_domains_ai2_adapt_dev_openmath_2_math_t4096_n500"

# Function to run training for a single method
run_training() {
    local METHOD=$1
    echo "----------------------------------------------------------------"
    echo "Running training for curriculum method: $METHOD"
    echo "----------------------------------------------------------------"
    
    # Run training with the specified curriculum method
    uv run python src/tinkering/tinker_openthoughts/train.py \
        dataset_name="$SOURCE" \
        curriculum_mode="$METHOD"
}

# Run trainings in parallel
run_training "${METHODS[0]}" &
run_training "${METHODS[1]}" &
run_training "${METHODS[2]}" &

wait

