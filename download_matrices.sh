#!/bin/bash
# Download the 18 representative benchmark matrices from SuiteSparse Matrix Collection
# These are the matrices from Table 2 of the TileSpGEMM paper (PPoPP '22)
# Run on login node (network I/O only, no computation)

set -euo pipefail

MATRIX_DIR="/scratch/general/vast/u1446071/spgemm_matrices"
mkdir -p "$MATRIX_DIR"

# SuiteSparse base URL
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# 18 representative matrices: Group/Name
MATRICES=(
    "DNVS/pdb1HYS"
    "Janna/consph"
    "Williams/cant"
    "Boeing/pwtk"
    "Hamm/scircuit"
    "Bova/rma10"
    "vanHeukelum/conf5_4-8x8-05"
    "DNVS/shipsec1"
    "Williams/mac_econ_fwd500"
    "Pajek/mc2depi"
    "Williams/cop20k_A"
    "Williams/webbase-1M"
    "Schenk_AFE/af_shell10"
    "Chen/pkustk12"
    "PARSEC/SiO2"
    "Zaoui/case39"
    "TSOPF/TSOPF_FS_b300_c2"
    "Janna/gupta3"
)

echo "Downloading ${#MATRICES[@]} matrices to $MATRIX_DIR"

for entry in "${MATRICES[@]}"; do
    GROUP=$(echo "$entry" | cut -d/ -f1)
    NAME=$(echo "$entry" | cut -d/ -f2)

    if [ -f "$MATRIX_DIR/$NAME.mtx" ]; then
        echo "[SKIP] $NAME already exists"
        continue
    fi

    echo -n "[DOWNLOAD] $NAME ... "
    URL="$BASE_URL/$GROUP/$NAME.tar.gz"
    if curl -sL "$URL" | tar xz -C "$MATRIX_DIR" 2>/dev/null; then
        # SuiteSparse tarballs extract to Name/Name.mtx — move up
        if [ -f "$MATRIX_DIR/$NAME/$NAME.mtx" ]; then
            mv "$MATRIX_DIR/$NAME/$NAME.mtx" "$MATRIX_DIR/$NAME.mtx"
            rm -rf "$MATRIX_DIR/$NAME"
        fi
        echo "OK ($(du -h "$MATRIX_DIR/$NAME.mtx" | cut -f1))"
    else
        echo "FAILED"
    fi
done

echo ""
echo "=== Downloaded matrices ==="
ls -lhS "$MATRIX_DIR"/*.mtx 2>/dev/null || echo "No matrices found!"
echo ""
echo "Total size: $(du -sh "$MATRIX_DIR" | cut -f1)"
