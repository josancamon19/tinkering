#!/bin/bash
set -e

MAX_TOKENS=${1:-4096}
LIMIT=${2:-500}

echo "Generating OpenThoughts datasets (max_tokens=$MAX_TOKENS, limit=$LIMIT)..."

# python -m tinkering.tinker_openthoughts.data.generator source=OPENMATH max_tokens=$MAX_TOKENS limit=$LIMIT
python -m tinkering.tinker_openthoughts.data.generator source=OPENCODE max_tokens=$MAX_TOKENS limit=$LIMIT
# python -m tinkering.tinker_openthoughts.data.generator source=ORGANIC_CHEM max_tokens=$MAX_TOKENS limit=$LIMIT
python -m tinkering.tinker_openthoughts.data.generator source=PHYSICS max_tokens=$MAX_TOKENS limit=$LIMIT
python -m tinkering.tinker_openthoughts.data.generator source=CODEGOLF max_tokens=$MAX_TOKENS limit=$LIMIT

echo "All datasets generated!"

# starts at 8.94 usd balance
# 276 seconds per eval, $0.18 per eval
# $0.65 per 20 steps of training
# 32 batch size, 5 epochs, 500 samples = 78 batches, should cost $3.4 per run. about $6 for 10 epochs, 2x batch size.