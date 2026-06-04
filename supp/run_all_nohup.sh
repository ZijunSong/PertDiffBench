#!/usr/bin/env bash
# Launch all supp gene-order experiments (reference only; use nohup commands below).
set -euo pipefail

SUPP="/data/ppnm/PertDiffBench/supp"
chmod +x "${SUPP}"/shuffle/*.sh "${SUPP}"/cluster/*.sh

echo "Step 0: generate reordered h5ad (once)"
echo "  cd /data/ppnm/PertDiffBench && python supp/preprocess_reorder_genes_cd4t.py --mode both"
echo
echo "Step 1: Batch A — shuffle (6 models on GPU 0-5, parallel)"
echo "Step 2: Batch B — cluster (same GPU mapping, run after Batch A finishes)"
echo
cat <<'EOF'
cd /data/ppnm/PertDiffBench
mkdir -p supp/logs/shuffle supp/logs/cluster

# ---------- Batch A: shuffle (6-way parallel) ----------
nohup bash supp/shuffle/ddpm_mlp.sh    > supp/logs/shuffle/ddpm_mlp.log    2>&1 &   # GPU 0
nohup bash supp/shuffle/ddpm.sh        > supp/logs/shuffle/ddpm.log        2>&1 &   # GPU 1
nohup bash supp/shuffle/scgen.sh       > supp/logs/shuffle/scgen.log       2>&1 &   # GPU 2
nohup bash supp/shuffle/scdiff.sh      > supp/logs/shuffle/scdiff.log      2>&1 &   # GPU 3
nohup bash supp/shuffle/squidiff.sh    > supp/logs/shuffle/squidiff.log    2>&1 &   # GPU 4
nohup bash supp/shuffle/scdiffusion.sh > supp/logs/shuffle/scdiffusion.log 2>&1 &   # GPU 5

# wait for Batch A, then:
# ---------- Batch B: cluster (6-way parallel, same GPU per model) ----------
nohup bash supp/cluster/ddpm_mlp.sh    > supp/logs/cluster/ddpm_mlp.log    2>&1 &   # GPU 0
nohup bash supp/cluster/ddpm.sh        > supp/logs/cluster/ddpm.log        2>&1 &   # GPU 1
nohup bash supp/cluster/scgen.sh       > supp/logs/cluster/scgen.log       2>&1 &   # GPU 2
nohup bash supp/cluster/scdiff.sh      > supp/logs/cluster/scdiff.log      2>&1 &   # GPU 3
nohup bash supp/cluster/squidiff.sh    > supp/logs/cluster/squidiff.log    2>&1 &   # GPU 4
nohup bash supp/cluster/scdiffusion.sh > supp/logs/cluster/scdiffusion.log 2>&1 &   # GPU 5
EOF
