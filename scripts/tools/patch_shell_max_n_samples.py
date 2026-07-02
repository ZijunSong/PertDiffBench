#!/usr/bin/env python3
"""One-shot patcher: unify shell scripts to compute max eval n_samples from test h5ad."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = 'source "scripts/lib/max_n_samples.sh"\n'

FIG1_TASK1_SAMPLES_BLOCK = re.compile(
    r"# n_samples per cell type\n"
    r"declare -A SAMPLES_MAP=\(\n"
    r"(?:  \['[^']+'\]=\d+\n)+"
    r"\)\n",
    re.MULTILINE,
)

FIG1_TASK1_REPLACE = (
    '# n_samples per cell type (max paired cells in valid set)\n'
    'source "scripts/lib/max_n_samples.sh"\n'
    'declare -A SAMPLES_MAP=()\n'
    'build_samples_map_from_valid_h5ad "${ROOT_DIR}" "${CELL_TYPES[@]}"\n'
)

FIG2_TASK2_SAMPLES_BLOCK = re.compile(
    r"declare -A SAMPLES_MAP=\(\n(?:  \[[^\]]+\]=\d+\n)+\)\n",
    re.MULTILINE,
)


def ensure_source(text: str) -> str:
    if "max_n_samples.sh" in text:
        return text
    # insert after first set -e line block
    m = re.search(r"(set -e[^\n]*\n(?:trap[^\n]*\n)?)", text)
    if m:
        return text[: m.end()] + "\n" + SOURCE + text[m.end() :]
    return SOURCE + text


def patch_fig1_task1(text: str) -> str:
    if "highly_variable_gene_gradient" not in text or "SAMPLES_MAP" not in text:
        return text
    return FIG1_TASK1_SAMPLES_BLOCK.sub(FIG1_TASK1_REPLACE, text, count=1)


def patch_fig2_task2_samples_map(text: str) -> str:
    if "fig2/task2" not in text and "fig2_task2" not in text:
        return text
    if "SAMPLES_MAP" not in text:
        return text

    def repl(m):
        return (
            'source "scripts/lib/max_n_samples.sh"\n'
            'declare -A SAMPLES_MAP=()\n'
            'for _ct in "${TARGET_CELL_TYPES[@]}"; do\n'
            '  _test="../../data/fig2/task2/task2_test_${_ct}_exp.h5ad"\n'
            '  SAMPLES_MAP["${_ct}"]="$(max_n_samples_paired "${_test}")"\n'
            'done\n'
        )

    return FIG2_TASK2_SAMPLES_BLOCK.sub(repl, text, count=1)


def inject_after_pattern(text: str, pattern: str, injection: str, flags=0) -> str:
    if injection.strip() in text:
        return text
    m = re.search(pattern, text, flags)
    if not m:
        return text
    return text[: m.end()] + injection + text[m.end() :]


def patch_fig2_task1_seed_loop(text: str) -> str:
    if "fig2/task1" not in text and "fig2_task1" not in text and "task1_unseen_pert" not in text:
        return text
    text = ensure_source(text)
    # After test_ds= or TEST_DATA= in seed loop
    for pat, inj in [
        (
            r'(test_ds="[^"]+")\n',
            '\n  N_SAMPLES="$(max_n_samples_paired "data/fig2/task1_unseen_pert/${test_ds}.h5ad")"\n',
        ),
        (
            r'(TEST_DATA="data/fig2/task1_unseen_pert/[^"]+")\n',
            '\n  N_SAMPLES="$(max_n_samples_paired "${TEST_DATA}")"\n',
        ),
        (
            r'(eval_dataset_name="[^"]+")\n',
            '\n  N_SAMPLES="$(max_n_samples_paired "data/fig2/task1_unseen_pert/${eval_dataset_name}.h5ad")"\n',
        ),
    ]:
        if "max_n_samples_paired" in text and "task1_unseen_pert" in text:
            break
        text = inject_after_pattern(text, pat, inj)
    text = re.sub(
        r'N_SAMPLES="\$\{N_SAMPLES:-\d+\}"\n',
        "",
        text,
    )
    text = re.sub(r"model\.params\.generation_kwargs\.n_samples=\d+", "model.params.generation_kwargs.n_samples=${N_SAMPLES}", text)
    text = re.sub(r'--num_samples\s+100\b', '--num_samples "${N_SAMPLES}"', text)
    text = re.sub(r'\$\{SAMPLES_MAP:-1000\}', '"${N_SAMPLES}"', text)
    return text


def patch_dataset_loop_valid_path(text: str, valid_var: str = "valid_data_path") -> str:
    """For scripts with per-dataset valid_data_path in loop."""
    inj = f'\n  N_SAMPLES="$(max_n_samples_paired "${{{valid_var}}}")"\n'
    if f'max_n_samples_paired "${{{valid_var}}}"' in text:
        return text
    text = ensure_source(text)
    text = re.sub(r"^N_SAMPLES=\d+.*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r'^N_SAMPLES="\$\{N_SAMPLES:-\d+\}".*\n', "", text, flags=re.MULTILINE)
    return inject_after_pattern(
        text,
        rf'({valid_var}="[^"]+")\n',
        inj,
    )


def patch_fig4(text: str) -> str:
    if "fig4" not in text.lower():
        return text
    text = ensure_source(text)
    text = re.sub(r'N_SAMPLES="\$\{N_SAMPLES:-\d+\}"', 'N_SAMPLES=""', text)
    if 'max_n_samples_timepoint' not in text and "fig4_test" in text:
        text = inject_after_pattern(
            text,
            r'(TEST_H5AD="[^"]+")\n',
            '\nN_SAMPLES="$(max_n_samples_timepoint "${TEST_H5AD}")"\n',
        )
    if 'max_n_samples_timepoint' not in text and 'fig4_test.h5ad' in text:
        text = inject_after_pattern(
            text,
            r'(TRAIN_H5="[^"]+fig4[^"]+")\n',
            '\nTEST_H5="${TRAIN_H5/fig4_train/fig4_test}"\nN_SAMPLES="$(max_n_samples_timepoint "${TEST_H5:-data/fig4_task1/fig4_test.h5ad}")"\n',
        )
    if "max_n_samples_timepoint" not in text:
        text = inject_after_pattern(
            text,
            r'(NUM_GENES=.*\n)',
            'source "scripts/lib/max_n_samples.sh"\n'
            'N_SAMPLES="$(max_n_samples_timepoint "data/fig4_task1/fig4_test.h5ad")"\n',
            flags=re.MULTILINE,
        )
    return text


def patch_noise(text: str) -> str:
    if "noise_exp" not in text:
        return text
    text = ensure_source(text)
    text = re.sub(r"--n_samples\s+6\b", '--n_samples "${N_SAMPLES}"', text)
    text = re.sub(r'N_SAMPLES:-6', 'N_SAMPLES:-', text)
    if "max_n_samples_paired" not in text and "valid_data_path" in text:
        text = patch_dataset_loop_valid_path(text)
    elif "max_n_samples_paired" not in text and "data-path" in text:
        text = inject_after_pattern(
            text,
            r'(valid_data_path="[^"]+")\n',
            '\n    N_SAMPLES="$(max_n_samples_paired "${valid_data_path}")"\n',
        )
    return text


def patch_file(path: Path) -> bool:
    text = path.read_text()
    orig = text
    text = patch_fig1_task1(text)
    text = patch_fig2_task2_samples_map(text)
    if "fig2/fig2_task1" in str(path) or "fig2_task1" in path.name:
        text = patch_fig2_task1_seed_loop(text)
    if "fig1/fig1_task2" in str(path) or "fig1_task2" in path.name:
        text = patch_dataset_loop_valid_path(text)
    if "fig1/fig1_task3" in str(path) or "fig1_task3" in path.name:
        text = patch_dataset_loop_valid_path(text)
    if "fig1/fig1_task4" in str(path) or "fig1_task4" in path.name:
        text = ensure_source(text)
        text = re.sub(r"declare -A SAMPLE_SIZES=\([^)]+\)", "declare -A SAMPLE_SIZES=()", text)
        if "build_sample_sizes" not in text:
            text = inject_after_pattern(
                text,
                r"(declare -A SAMPLE_SIZES=\(\)\n)",
                (
                    "source \"scripts/lib/max_n_samples.sh\"\n"
                    "# SAMPLE_SIZES filled per dataset from test h5ad in loop\n"
                ),
            )
    if "fig2/fig2_task3" in str(path) or "fig2_task3" in path.name:
        text = ensure_source(text)
        text = re.sub(r'N_SAMPLES="\$\{N_SAMPLES:-\d+\}"', "", text)
        text = re.sub(r"--n_samples\s+1000\b", '--n_samples "${N_SAMPLES}"', text)
        text = re.sub(r'--num_samples\s+100\b', '--num_samples "${N_SAMPLES}"', text)
        if "max_n_samples_paired" not in text:
            text = inject_after_pattern(
                text,
                r'(TEST_H5AD="[^"]+")\n',
                '\n  N_SAMPLES="$(max_n_samples_paired "${TEST_H5AD}")"\n',
            )
    if "fig2_task2" in str(path) or "fig2_task2_plus" in str(path) or "fig2_task2_extend" in str(path):
        text = ensure_source(text)
        text = re.sub(r'N_SAMPLES="\$\{N_SAMPLES:-\d+\}"', "", text)
        text = re.sub(r'NUM_SAMPLES="\$\{NUM_SAMPLES:-\d+\}"', "", text)
        text = re.sub(r"generation_kwargs\.n_samples=\d+", "generation_kwargs.n_samples=${N_SAMPLES}", text)
        text = re.sub(r'--num_samples\s+100\b', '--num_samples "${N_SAMPLES}"', text)
    if "fig2_task1_moa" in str(path) or "moa" in path.name:
        text = ensure_source(text)
        text = re.sub(r'N_SAMPLES="\$\{N_SAMPLES:-\d+\}"', "", text)
        text = re.sub(r'NUM_SAMPLES="\$\{NUM_SAMPLES:-\d+\}"', "", text)
        text = re.sub(r"generation_kwargs\.n_samples=\d+", "generation_kwargs.n_samples=${N_SAMPLES}", text)
        if "max_n_samples_paired" not in text and "TEST_H5AD" in text:
            text = inject_after_pattern(
                text,
                r'(TEST_H5AD="[^"]+")\n',
                '\n  N_SAMPLES="$(max_n_samples_paired "${TEST_H5AD}")"\n',
            )
        if "max_n_samples_paired" not in text and "test_h5ad" in text:
            text = inject_after_pattern(
                text,
                r'(test_h5ad="[^"]+")\n',
                '\n  N_SAMPLES="$(max_n_samples_paired "${test_h5ad}")"\n',
            )
    if "fig4" in str(path):
        text = patch_fig4(text)
    if "noise_exp" in str(path):
        text = patch_noise(text)
    if "encoder_exp" in str(path) and path.suffix == ".sh":
        text = ensure_source(text)
        text = re.sub(r'N_SAMPLES="\$\{N_SAMPLES:-\d+\}"', "", text)
    if "highly_variable_gene_gradient" in str(path):
        text = ensure_source(text)
        text = re.sub(r"--n_samples\s+\d+\b", '--n_samples "${N_SAMPLES}"', text)
        if "max_n_samples_paired" not in text:
            text = inject_after_pattern(
                text,
                r'(valid_path="[^"]+")\n',
                '\n  N_SAMPLES="$(max_n_samples_paired "${valid_path}")"\n',
            )
    if text != orig:
        path.write_text(text)
        return True
    return False


def main() -> None:
    changed = []
    for base in [REPO / "scripts", REPO / "supp"]:
        if not base.exists():
            continue
        for p in base.rglob("*.sh"):
            if patch_file(p):
                changed.append(p.relative_to(REPO))
    print(f"Patched {len(changed)} shell scripts")
    for p in sorted(changed)[:30]:
        print(" ", p)
    if len(changed) > 30:
        print(f"  ... and {len(changed)-30} more")


if __name__ == "__main__":
    main()
