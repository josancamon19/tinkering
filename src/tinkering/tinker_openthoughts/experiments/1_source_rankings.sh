# Explicitly list the datasets to process
SOURCES=(
    "openthoughts_all_domains_ai2_adapt_dev_openmath_2_math_t4096_n500"
    # "openthoughts_all_domains_nvidia_OpenCodeReasoning_t4096_n500"
    # "openthoughts_all_domains_organic_chemistry_questions_t4096_n500"
    # "openthoughts_all_domains_stackexchange_codegolf_t4096_n500"
    # "openthoughts_all_domains_stackexchange_physics_t4096_n500"
)

# Function to run training for a single source
run_training() {
    local SOURCE=$1
    echo "----------------------------------------------------------------"
    echo "Running training for source: $SOURCE"
    echo "----------------------------------------------------------------"
    
    # Run training with default parameters, overriding only dataset_name and wandb_group
    uv run python src/tinkering/tinker_openthoughts/train.py dataset_name="$SOURCE" pass_at_k=1 full_parallel=True
}

run_training "${SOURCES[0]}" &
# run_training "${SOURCES[1]}" &
# run_training "${SOURCES[2]}" &
# run_training "${SOURCES[3]}" &
# run_training "${SOURCES[4]}" &
wait
