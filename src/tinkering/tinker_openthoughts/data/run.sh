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