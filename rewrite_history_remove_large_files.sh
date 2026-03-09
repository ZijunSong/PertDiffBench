#!/bin/bash
# 从 Git 历史中彻底移除大文件，使 push 不再触发 GitHub 100MB 限制
set -e
cd /share/PertBench

echo "1. 移除可能存在的 index 锁..."
rm -f .git/index.lock

echo "2. 切回 main 分支..."
git checkout main 2>/dev/null || true

echo "3. 从所有历史提交中移除大文件（可能需要几分钟）..."
export FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch --force --index-filter '
  git rm --cached --ignore-unmatch \
    src/Geneformer/model.safetensors \
    src/scFoundation/preprocessing/preprocessed_task1_train_CD4T_exp.h5ad \
    src/scFoundation/preprocessing/preprocessed_task1_valid_CD4T_exp.h5ad \
    src/scFoundation/preprocessing/task1_train_CD4T_exp.h5ad \
    "src/scFoundation/preprocessing/encoder_exp/scfoundation_ddpm/preprocessed_task1_valid_CD4T_exp.h5ad" \
    src/scDiffusion/results/scdiffusion/synthetic_prediction.h5ad \
    2>/dev/null || true
' --prune-empty main

echo "4. 清理备份引用..."
rm -rf .git/refs/original/

echo "5. 运行 gc 回收大对象..."
git reflog expire --expire=now --all && git gc --prune=now --aggressive

echo "完成。请执行: git push -u origin main --force"
echo "（若仍报大文件，可先运行: git rev-list --objects main | git cat-file --batch-check=\"%(objecttype) %(objectname) %(objectsize) %(rest)\" | sed -n \"s/^blob //p\" | sort -rnk2 | head -5 检查是否还有大 blob）"
