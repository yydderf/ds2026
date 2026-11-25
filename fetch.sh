#!/usr/bin/env bash
# Fetch datasets for the project.
#   Part 1: WANDS (git submodule at part1/WANDS)
#   Part 2: H&M dataset from qdrant/ann-filtering-benchmark-datasets
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------- Part 1: WANDS ----------
WANDS_DIR="part1/WANDS/dataset"
WANDS_FILES=(product.csv query.csv label.csv)

wands_ok() {
    for f in "${WANDS_FILES[@]}"; do
        [[ -s "$WANDS_DIR/$f" ]] || return 1
    done
}

echo "==> Part 1: checking WANDS dataset"
if wands_ok; then
    echo "    WANDS dataset present."
else
    echo "    WANDS dataset missing, initializing submodule..."
    git submodule update --init --recursive part1/WANDS
    if wands_ok; then
        echo "    WANDS dataset fetched."
    else
        echo "    ERROR: WANDS dataset still missing in $WANDS_DIR" >&2
        exit 1
    fi
fi

# ---------- Part 2: H&M ----------
HNM_URL="https://storage.googleapis.com/ann-filtered-benchmark/datasets/hnm.tgz"
HNM_DIR="part2/hnm"
HNM_FILES=(vectors.npy payloads.jsonl tests.jsonl filters.json)

hnm_ok() {
    for f in "${HNM_FILES[@]}"; do
        [[ -s "$HNM_DIR/$f" ]] || return 1
    done
}

echo "==> Part 2: checking H&M dataset"
if hnm_ok; then
    echo "    H&M dataset present in $HNM_DIR."
else
    echo "    Downloading H&M dataset (~950 MB)..."
    mkdir -p "$HNM_DIR"
    TMP="$HNM_DIR/hnm.tgz.part"
    # -C - resumes an interrupted download
    curl -fL --retry 3 -C - -o "$TMP" "$HNM_URL"
    echo "    Extracting..."
    tar -xzf "$TMP" -C "$HNM_DIR"
    rm -f "$TMP"
    if hnm_ok; then
        echo "    H&M dataset ready in $HNM_DIR."
    else
        echo "    ERROR: H&M dataset incomplete in $HNM_DIR" >&2
        exit 1
    fi
fi

echo "==> All datasets ready."
