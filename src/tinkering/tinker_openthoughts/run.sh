uv run python src/tinkering/tinker_openthoughts/train.py dataset_name=openthoughts_all_domains_ai2_adapt_dev_openmath_2_math_t4096_n500 epochs=1 pass_at_k=5 &
sleep 2
uv run python src/tinkering/tinker_openthoughts/train.py dataset_name=openthoughts_all_domains_ai2_adapt_dev_openmath_2_math_t4096_n500 epochs=1 pass_at_k=5 &
wait