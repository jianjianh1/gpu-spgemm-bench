#!/bin/bash
# Download all candidate SpGEMM benchmark matrices from SuiteSparse.
# Uses datasets/mat_candidates.csv (238 square matrices with estimated flops >= 2e8).
# The TileSpGEMM paper uses 142 of these (exact flop threshold >= 1e9).
# Run on login node (network I/O only, no computation).

set -euo pipefail

MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"
CSV="$(dirname "$0")/datasets/mat_candidates.csv"
BASE_URL="https://sparse.tamu.edu/MM"

mkdir -p "$MATRIX_DIR"

if [ ! -f "$CSV" ]; then
    echo "ERROR: $CSV not found. Run the Python script to generate it first."
    exit 1
fi

TOTAL=$(tail -n +2 "$CSV" | wc -l)
DONE=0
SKIP=0
FAIL=0

echo "Downloading up to $TOTAL matrices to $MATRIX_DIR"
echo ""

tail -n +2 "$CSV" | while IFS=',' read -r GROUP NAME ROWS COLS NNZ EST_FLOPS; do
    DONE=$((DONE + 1))

    if [ -f "$MATRIX_DIR/$NAME.mtx" ]; then
        SKIP=$((SKIP + 1))
        continue
    fi

    echo -n "[$DONE/$TOTAL] $NAME ($GROUP) ... "
    URL="$BASE_URL/$GROUP/$NAME.tar.gz"
    if curl -sL --max-time 120 "$URL" | tar xz -C "$MATRIX_DIR" 2>/dev/null; then
        if [ -f "$MATRIX_DIR/$NAME/$NAME.mtx" ]; then
            mv "$MATRIX_DIR/$NAME/$NAME.mtx" "$MATRIX_DIR/$NAME.mtx"
            rm -rf "$MATRIX_DIR/$NAME"
        fi
        SIZE=$(du -h "$MATRIX_DIR/$NAME.mtx" | cut -f1)
        echo "OK ($SIZE)"
    else
        echo "FAILED"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Download complete ==="
DOWNLOADED=$(ls "$MATRIX_DIR"/*.mtx 2>/dev/null | wc -l)
echo "Matrices on disk: $DOWNLOADED"
echo "Total size: $(du -sh "$MATRIX_DIR" | cut -f1)"
