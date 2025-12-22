#!/bin/bash
# Generate OpenThoughts datasets for each source
# max_tokens=4096, limit=1000

set -e

echo "=========================================="
echo "OpenThoughts Dataset Generation"
echo "=========================================="
echo ""

# OpenMath
echo "[1/5] Generating OpenMath dataset..."
python -m tinkering.tinker_openthoughts.datasets_gen \
    source=OPENMATH \
    max_tokens=4096 \
    limit=1000

echo ""

# OpenCode
echo "[2/5] Generating OpenCode dataset..."
python -m tinkering.tinker_openthoughts.datasets_gen \
    source=OPENCODE \
    max_tokens=4096 \
    limit=1000

echo ""

# Organic Chemistry
echo "[3/5] Generating Organic Chemistry dataset..."
python -m tinkering.tinker_openthoughts.datasets_gen \
    source=ORGANIC_CHEM \
    max_tokens=4096 \
    limit=1000

echo ""

# Physics
echo "[4/5] Generating Physics dataset..."
python -m tinkering.tinker_openthoughts.datasets_gen \
    source=PHYSICS \
    max_tokens=4096 \
    limit=1000

echo ""

# Code Golf
echo "[5/5] Generating Code Golf dataset..."
python -m tinkering.tinker_openthoughts.datasets_gen \
    source=CODEGOLF \
    max_tokens=4096 \
    limit=1000

echo ""
echo "=========================================="
echo "All datasets generated successfully!"
echo "=========================================="

