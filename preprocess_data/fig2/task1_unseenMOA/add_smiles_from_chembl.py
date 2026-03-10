#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Iterable, Set, List

import anndata as ad
import pandas as pd
import requests


CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
# 处理后数据的根路径（可从项目根目录运行脚本）
DATA_OUT = Path("/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA")
# cache 放在脚本所在目录，便于复用
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = str(_SCRIPT_DIR / "chembl_smiles_cache.json")


def load_cache(cache_path: Path) -> Dict[str, Optional[str]]:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # obj: {CHEMBLxxxx: "SMILES" or null}
        return {k: v for k, v in obj.items()}
    return {}


def save_cache(cache_path: Path, cache: Dict[str, Optional[str]]) -> None:
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(cache_path)


def fetch_smiles_one(chembl_id: str, timeout: int = 30) -> Optional[str]:
    """
    Query ChEMBL molecule endpoint:
      GET /chembl/api/data/molecule/{CHEMBL_ID}.json
    Read molecule_structures.canonical_smiles.
    """
    url = f"{CHEMBL_API}/molecule/{chembl_id}.json"
    r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    if r.status_code != 200:
        return None
    data = r.json()
    ms = data.get("molecule_structures") or {}
    # canonical_smiles is most commonly used
    return ms.get("canonical_smiles")


def get_smiles_with_retry(
    chembl_id: str,
    cache: Dict[str, Optional[str]],
    max_retries: int = 4,
    sleep_base: float = 0.8,
) -> Optional[str]:
    if chembl_id in cache:
        return cache[chembl_id]

    last_err = None
    for attempt in range(max_retries):
        try:
            smi = fetch_smiles_one(chembl_id)
            cache[chembl_id] = smi
            return smi
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt))
    # record failure as None to avoid infinite retries across runs
    cache[chembl_id] = None
    return None


def parse_chembl_cell_value(v) -> List[str]:
    """
    Turn an obs['chembl.ID'] cell value into a list of CHEMBL IDs.
    Handles:
      - NaN/None -> []
      - "CHEMBL1" -> ["CHEMBL1"]
      - "CHEMBL1;CHEMBL2" -> ["CHEMBL1","CHEMBL2"]
      - weird whitespace
    """
    if v is None:
        return []
    if isinstance(v, float) and pd.isna(v):
        return []
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    # Basic sanity: only keep tokens starting with CHEMBL
    ids = [p for p in parts if p.upper().startswith("CHEMBL")]
    return ids


def collect_unique_chembl_ids(h5ad_path: Path, col: str) -> Set[str]:
    adata = ad.read_h5ad(h5ad_path)
    if col not in adata.obs.columns:
        return set()
    ids: Set[str] = set()
    for v in adata.obs[col].tolist():
        ids.update(parse_chembl_cell_value(v))
    return ids


def add_smiles_column_to_one(
    h5ad_in: Path,
    h5ad_out: Path,
    cache: Dict[str, Optional[str]],
    chembl_col: str = "chembl.ID",
    smiles_col: str = "smiles",
    max_retries: int = 4,
) -> None:
    adata = ad.read_h5ad(h5ad_in)

    if chembl_col not in adata.obs.columns:
        raise KeyError(f"Column '{chembl_col}' not found in obs for {h5ad_in}")

    if smiles_col in adata.obs.columns:
        # keep idempotent; still overwrite if user wants, but default: skip
        print(f"[SKIP] {h5ad_in.name}: obs already has '{smiles_col}'")
        adata.write(h5ad_out)
        return

    # Build smiles per row
    smiles_values: List[Optional[str]] = []
    for v in adata.obs[chembl_col].tolist():
        ids = parse_chembl_cell_value(v)
        if not ids:
            smiles_values.append(None)
            continue
        smi_list: List[str] = []
        for cid in ids:
            smi = get_smiles_with_retry(cid, cache=cache, max_retries=max_retries)
            smi_list.append(smi if smi is not None else "")
        # join back; keep position aligned with IDs
        smiles_values.append(";".join(smi_list))

    adata.obs[smiles_col] = pd.Series(smiles_values, index=adata.obs_names, dtype="object")
    adata.uns["smiles_source"] = {
        "chembl_api": CHEMBL_API,
        "chembl_id_col": chembl_col,
        "smiles_col": smiles_col,
        "note": "SMILES fetched from ChEMBL molecule endpoint (canonical_smiles).",
    }

    h5ad_out.parent.mkdir(parents=True, exist_ok=True)
    adata.write(h5ad_out)
    print(f"[OK] wrote {h5ad_out} (added obs['{smiles_col}'])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=str(DATA_OUT),
        help="root dir containing control_plus_ifn/ (default: DATA_OUT)",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="control_plus_ifn/**/*.h5ad",
        help="which h5ad files to process (relative to root)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="control_plus_ifn_with_smiles",
        help="output directory under root (mirrors structure)",
    )
    ap.add_argument(
        "--chembl_col",
        type=str,
        default="chembl.ID",
        help="obs column containing ChEMBL IDs",
    )
    ap.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="obs column to create",
    )
    ap.add_argument(
        "--cache",
        type=str,
        default=DEFAULT_CACHE,
        help="JSON cache file path (default: script dir / chembl_smiles_cache.json)",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=4,
        help="max retries per ChEMBL ID",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cache_path = Path(args.cache).resolve() if Path(args.cache).is_absolute() else root / args.cache
    cache = load_cache(cache_path)

    # Find files
    files = sorted((root).glob(args.glob))
    # Exclude control_merged.h5ad by default; it has no drug anyway
    files = [p for p in files if p.is_file() and p.name != "control_merged.h5ad"]

    if not files:
        raise FileNotFoundError(f"No h5ad matched: root={root} glob={args.glob}")

    print(f"[INFO] matched {len(files)} h5ad files")
    print(f"[INFO] cache: {cache_path} (existing keys={len(cache)})")

    # Pre-collect unique IDs to warm cache (fewer repeated network hits)
    all_ids: Set[str] = set()
    for fp in files:
        all_ids |= collect_unique_chembl_ids(fp, args.chembl_col)
    all_ids = {cid for cid in all_ids if cid}  # safety

    print(f"[INFO] unique ChEMBL IDs needed: {len(all_ids)}")
    # Fetch missing IDs
    miss = [cid for cid in sorted(all_ids) if cid not in cache]
    if miss:
        print(f"[INFO] fetching missing IDs: {len(miss)}")
    for idx, cid in enumerate(miss, 1):
        smi = get_smiles_with_retry(cid, cache=cache, max_retries=args.max_retries)
        if idx % 50 == 0:
            save_cache(cache_path, cache)
            print(f"[INFO] cache saved ({idx}/{len(miss)})")
        time.sleep(0.05)  # be polite to the API

    save_cache(cache_path, cache)

    out_root = root / args.out_dir
    for fp in files:
        rel = fp.relative_to(root)
        out_fp = out_root / rel
        if out_fp.exists():
            print(f"[SKIP] exists: {out_fp}")
            continue
        add_smiles_column_to_one(
            h5ad_in=fp,
            h5ad_out=out_fp,
            cache=cache,
            chembl_col=args.chembl_col,
            smiles_col=args.smiles_col,
            max_retries=args.max_retries,
        )
        save_cache(cache_path, cache)

    print("[DONE]")


if __name__ == "__main__":
    main()
