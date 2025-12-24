# Tinkering with OpenThoughts 🧠

A streamlined environment for exploring, subsetting, and training on the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset. Built with [tinker](https://thinkingmachines.ai/tinker/) for infrastructure and evaluation, replacing the original LlamaFactory + Evalchemy stack from the [open-thoughts repo](https://github.com/open-thoughts/open-thoughts).

> 📄 Based on the [OpenThoughts paper](https://arxiv.org/abs/2506.04178): *"OpenThoughts: Data Recipes for Reasoning Models"*

---

## 🎬 Demo

[**→ Live Dataset Explorer: openthoughts.streamlit.app**](https://openthoughts.streamlit.app)

<!-- VIDEO_PLACEHOLDER: Add demo video here -->

<!-- TWITTER_PLACEHOLDER: Add link to Twitter/X post -->

---

## Why This Repo?

[Tinker](https://thinkingmachines.ai/tinker/) is excellent infrastructure for ML training—async pipelined loops, parallelized evals, simple checkpointing. This repo makes it easy for anyone to run dataset ablations and experiments on OpenThoughts (one of the best open source datasets) without wrestling with complex setups.

**One command to train. One command to generate subsets. Evals run in parallel.**

> ⚠️ **Note on evaluation:** The original paper uses pass@1 with deterministic inference. Since tinker doesn't yet support deterministic mode, we use **pass@7** to compensate for run-to-run variance.

---

## 🧪 Ablations & Experiments

This repo is designed for running sweeps across multiple dimensions. Mix and match to find your optimal data recipe.

### Data Sources

Compare performance across different question sources from the OpenThoughts3-1.2M mix:

| Source | Domain | Description |
|--------|--------|-------------|
| `ai2-adapt-dev/openmath-2-math` | Math | High-quality math problems |
| `nvidia/OpenCodeReasoning` | Code | Competitive programming & algorithms |
| `organic-chemistry-questions` | Science | Chemistry reasoning |
| `stackexchange-physics` | Science | Physics Q&A |
| `stackexchange_codegolf` | Code | Code golf challenges |

```bash
# Run source comparison sweep (parallel)
bash src/tinkering/tinker_openthoughts/experiments/1_source_rankings.sh
```

### Domains

Filter by high-level domain: `code`, `math`, `science`, or `all`.

### Token Lengths

Control complexity by filtering max conversation length. Shorter = faster iteration, longer = harder problems.

```bash
# Quick experiments (shorter sequences)
uv run python src/tinkering/tinker_openthoughts/data/generator.py max_tokens=2048 limit=200

# Full complexity
uv run python src/tinkering/tinker_openthoughts/data/generator.py max_tokens=8192 limit=500
```

### Curriculum Learning

Order training data by difficulty. Requires datasets with `difficulty` annotations.

| Mode | Description |
|------|-------------|
| `none` | Default random order |
| `easy_to_hard` | All epochs sorted by ascending difficulty |
| `first_epoch_only` | First epoch sorted, rest shuffled |
| `grouped_shuffle` | Group by difficulty, shuffle within groups, concat easy→hard |

```bash
# Run curriculum sweep (parallel)
bash src/tinkering/tinker_openthoughts/experiments/2_curriculum_methods.sh

# Or manually
uv run python src/tinkering/tinker_openthoughts/train.py \
    dataset_name="..." \
    curriculum_mode="grouped_shuffle"
```

### Example Sweep

Run multiple configurations in parallel:

```bash
# Terminal 1: Math data, easy-to-hard curriculum
uv run python src/tinkering/tinker_openthoughts/train.py \
    dataset_name="openthoughts_all_domains_ai2_adapt_dev_openmath_2_math_t4096_n500" \
    curriculum_mode="easy_to_hard" &

# Terminal 2: Code data, no curriculum
uv run python src/tinkering/tinker_openthoughts/train.py \
    dataset_name="openthoughts_all_domains_nvidia_OpenCodeReasoning_t4096_n500" \
    curriculum_mode="none" &

wait
```

---

## 🚀 Workflow

### Step 1: Generate a Dataset Subset

Use the generator to create filtered subsets of OpenThoughts3-1.2M. Filter by source (e.g., `nvidia/OpenCodeReasoning`), domain, or max token length.

```bash
uv run python src/tinkering/tinker_openthoughts/data/generator.py \
    source=Source.OPENMATH \
    max_tokens=4096 \
    limit=500
```

This creates a folder in `subsets/` with:
- `dataset/` — HuggingFace-compatible dataset ready for training
- `metadata.json` — Filter config and stats
- `samples/` — 10 sample `.md` files to quickly inspect thinking traces and outputs

### Step 2: Inspect Your Subset

Browse the generated `samples/*.md` files to quickly verify quality—each file shows the user input, thinking trace, and final output in a readable format.

> 💡 The [Streamlit explorer](https://openthoughts.streamlit.app) is for browsing the **full OpenThoughts3-1.2M dataset**, not your generated subsets.

### Step 3: Train

Run training with your subset. All hyperparameters are configurable via CLI:

```bash
uv run python src/tinkering/tinker_openthoughts/train.py \
    dataset_name="openthoughts_all_domains_ai2_adapt_dev_openmath_2_math_t4096_n500" \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    batch_size=32 \
    learning_rate=1e-5 \
    epochs=5 \
    lora_rank=32
```

---

## ⚙️ Hyperparameters

The default hyperparameters are tuned based on the [OpenThoughts paper](https://arxiv.org/abs/2506.04178) recommendations, with additional ablations run on this codebase:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `batch_size` | 32 | Ablated; 32 was best for stability and throughput |
| `learning_rate` | 1e-5 | Standard for LoRA fine-tuning on reasoning tasks |
| `epochs` | 5 | Paper suggests 1-5 depending on dataset size |
| `lora_rank` | 32 | Matches paper; rank 16 works but 32 is safer |
| `train_split` | 0.9 | 90% train, 10% held-out for NLL eval |

Curriculum learning modes are also available (`easy_to_hard`, `grouped_shuffle`, etc.) for datasets with difficulty annotations.

---

## 📊 Evaluation

Evaluations run **in parallel** during training without blocking the training loop. Metrics are logged to W&B.

| Benchmark | Frequency | Description |
|-----------|-----------|-------------|
| **NLL (test split)** | Every 5 steps | Negative log-likelihood on held-out data |
| **AIME 2025** | Every 10 steps | Math olympiad problems (30 samples) |
| **GPQA** | Every 10 steps | Graduate-level science QA (50 samples) |
| **LiveCodeBench** | Every 10 steps | Competitive programming (30 samples) |

All benchmarks use **pass@7** to account for tinker's non-deterministic inference.

---

## 🏗 Project Structure

```
src/tinkering/
├── tinker_openthoughts/          # Main training code
│   ├── train.py                  # Training loop with tinker
│   ├── common.py                 # Utilities and helpers
│   ├── data/
│   │   └── generator.py          # Subset generation tool
│   ├── evals/                    # Evaluation implementations
│   │   ├── aime.py
│   │   ├── gpqad.py
│   │   ├── livecodebench.py
│   │   └── nll.py
│   └── experiments/              # Curated experiment scripts
│       ├── 1_source_rankings.sh
│       ├── 2_curriculum_methods.sh
│       └── 3_mixtures.sh
│
└── exploring_openthoughts/       # Full dataset exploration (powers openthoughts.streamlit.app)
    ├── main.py                   # Streamlit dashboard for OpenThoughts3-1.2M
    ├── stats.py                  # Pre-compute statistics
    └── filters.py                # Compute filter options
```

---

## 🔗 References

- [OpenThoughts Paper (arXiv)](https://arxiv.org/abs/2506.04178)
- [OpenThoughts GitHub](https://github.com/open-thoughts/open-thoughts)
- [OpenThoughts3-1.2M Dataset](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- [Tinker Documentation](https://thinkingmachines.ai/tinker/)
