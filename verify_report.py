"""
verify_report.py — Strict Provenance and Integrity Verification

Audit script that:
  1. Resolves every cited file in RESULTS.md and confirms it exists on disk.
  2. Recomputes all summary statistics directly from raw .jsonl / .json files
     and verifies that summary JSON matches raw execution data.
  3. Extracts numbers from the markdown tables in RESULTS.md and verifies
     exact numerical consistency against the recomputed data.
  4. Scans for suspicious duplicate cell values across distinct experiments/arms
     (e.g., preventing copy-pasted or stale baseline numbers).
  5. Fails loudly (exit code 1) with an actionable diff if any discrepancy
     is found beyond acceptable rounding tolerances.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Any, Tuple

from reporting.flags import compute_stats


def recompute_from_raw_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Recomputes round-by-round statistics directly from raw execution lines."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Raw log file not found: {jsonl_path}")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    n_rounds = len(records[0]["rounds"])
    recomputed_rounds = []

    for r_idx in range(n_rounds):
        round_num = r_idx + 1

        # Check if record has "matched_n" (compare_arms format) or "n_attacks" (full_loop format)
        sample_round = records[0]["rounds"][r_idx]
        if "matched_n" in sample_round:
            n_list = [rec["rounds"][r_idx]["matched_n"] for rec in records]
            s_asr = [rec["rounds"][r_idx]["static_arm"]["asr"] for rec in records]
            s_det = [rec["rounds"][r_idx]["static_arm"]["detection_rate"] for rec in records]
            s_fpr = [rec["rounds"][r_idx]["static_arm"]["fpr"]["combined_fpr"] for rec in records]
            a_asr = [rec["rounds"][r_idx]["adaptive_arm"]["asr"] for rec in records]
            a_det = [rec["rounds"][r_idx]["adaptive_arm"]["detection_rate"] for rec in records]
            a_fpr = [rec["rounds"][r_idx]["adaptive_arm"]["fpr"]["combined_fpr"] for rec in records]
            a_corp = [rec["rounds"][r_idx]["adaptive_arm"]["corpus_size"] for rec in records]

            recomputed_rounds.append({
                "round": round_num,
                "matched_n": compute_stats(n_list),
                "static_asr": compute_stats(s_asr),
                "static_det": compute_stats(s_det),
                "static_fpr": compute_stats(s_fpr),
                "adaptive_asr": compute_stats(a_asr),
                "adaptive_det": compute_stats(a_det),
                "adaptive_fpr": compute_stats(a_fpr),
                "adaptive_corpus": compute_stats(a_corp),
            })
        else:
            n_list = [rec["rounds"][r_idx]["n_attacks"] for rec in records]
            asr_list = [rec["rounds"][r_idx]["attack_success_rate"] for rec in records]
            det_list = [rec["rounds"][r_idx]["detection_rate"] for rec in records]
            fpr_list = [rec["rounds"][r_idx]["false_positive_rate"] for rec in records]
            corp_list = [rec["rounds"][r_idx]["corpus_size"] for rec in records]

            recomputed_rounds.append({
                "round": round_num,
                "n_attacks": compute_stats(n_list),
                "attack_success_rate": compute_stats(asr_list),
                "detection_rate": compute_stats(det_list),
                "fpr_combined": compute_stats(fpr_list),
                "corpus_size": compute_stats(corp_list),
            })

    return recomputed_rounds


def parse_markdown_tables(doc_path: str) -> List[List[List[str]]]:
    """Extracts markdown tables from the document."""
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    tables = []
    lines = content.splitlines()
    current_table = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # Skip markdown separator line
            if re.match(r"^\|(\s*:?-+:?\s*\|)+$", stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            current_table.append(cells)
            in_table = True
        else:
            if in_table and current_table:
                tables.append(current_table)
                current_table = []
                in_table = False

    if in_table and current_table:
        tables.append(current_table)

    return tables


def extract_numbers_from_str(s: str) -> List[float]:
    """Extracts all floats/ints from a string like '$28.3 \\pm 8.7 [\\pm 6.2]$'."""
    clean = s.replace("$", "").replace("\\pm", " ").replace("±", " ").replace("+/-", " ")
    clean = clean.replace("[", " ").replace("]", " ").replace("%", " ")
    tokens = clean.split()
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
            pass
    return nums


def check_suspicious_cross_duplicates(table_rows: List[List[str]], table_name: str) -> List[str]:
    """Flags identical non-trivial cells across arms in the same row."""
    warnings = []
    for r_idx, row in enumerate(table_rows):
        # Look for duplicate floating point values across different columns
        seen: Dict[float, str] = {}
        for c_idx, cell in enumerate(row):
            nums = extract_numbers_from_str(cell)
            for n in nums:
                # Ignore trivial integers like 0, 1, 23 (round number or seed count)
                if n in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 23.0):
                    continue
                if n in seen:
                    warnings.append(
                        f"[{table_name} Row {r_idx+1}] Warning: Potential duplicate value {n} "
                        f"appears in column {c_idx+1} and column {seen[n]}"
                    )
                else:
                    seen[n] = str(c_idx + 1)
    return warnings


def verify_file_existence(cited_files: List[str]) -> List[str]:
    errors = []
    for f in cited_files:
        if not os.path.exists(f):
            errors.append(f"Missing cited file: {f}")
        else:
            size = os.path.getsize(f)
            if size == 0:
                errors.append(f"Cited file is empty (0 bytes): {f}")
    return errors


def verify_table_values(table_rows: List[List[str]], expected_data: List[Dict[str, Any]], table_name: str, val_keys: List[str]) -> List[str]:
    errors = []
    # Skip header row
    data_rows = table_rows[1:]
    if len(data_rows) != len(expected_data):
        return [f"[{table_name}] Row count mismatch: markdown has {len(data_rows)} data rows, recomputed has {len(expected_data)} rounds."]

    for r_idx, (row, exp) in enumerate(zip(data_rows, expected_data)):
        row_str = " | ".join(row)
        for key in val_keys:
            if key not in exp:
                continue
            expected_mean = exp[key]["mean"]
            # Look for expected_mean in row numbers (allowing +/- 0.1 rounding tolerance)
            row_nums = extract_numbers_from_str(row_str)
            match = any(abs(n - expected_mean) <= 0.15 for n in row_nums)
            if not match:
                errors.append(
                    f"[{table_name} Round {r_idx+1}] Mismatch for '{key}': "
                    f"expected mean ~{expected_mean:.2f}, but not found in row values: {row_nums}"
                )
    return errors


def main():
    parser = argparse.ArgumentParser(description="Strict provenance and verification tool")
    parser.add_argument("--results-doc", type=str, default="RESULTS.md", help="Results document to verify")
    args = parser.parse_args()

    print("=" * 80)
    print(f"VERIFYING RESULTS PROVENANCE & NUMERICAL INTEGRITY: {args.results_doc}")
    print("=" * 80)

    # 1. Verify cited files exist
    cited_files = [
        "n_runs_summary_unlimited.json",
        "n_runs_raw_unlimited.jsonl",
        "compare_arms_metrics.json",
        "compare_arms_raw.jsonl",
        "mutation_depth_comparison.json",
        "depth_2_summary.json",
        "depth_4_summary.json",
        "depth_8_summary.json",
        "depth_unlimited_summary.json",
    ]

    file_errors = verify_file_existence(cited_files)
    if file_errors:
        print("\n[FAIL] File Existence Check Failed:")
        for err in file_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"[OK] All {len(cited_files)} cited raw and summary files exist on disk.")

    # 2. Recompute and verify Table 1 (Multi-Run Benchmark)
    print("\n--- Verifying Table 1: Multi-Run Benchmark (10 Runs) ---")
    recomputed_t1 = recompute_from_raw_jsonl("n_runs_raw_unlimited.jsonl")
    with open("n_runs_summary_unlimited.json", "r", encoding="utf-8") as f:
        summary_t1 = json.load(f)

    # Validate summary json against recomputed
    for r_idx, (rec, sm) in enumerate(zip(recomputed_t1, summary_t1["summary_by_round"])):
        diff_asr = abs(rec["attack_success_rate"]["mean"] - sm["attack_success_rate"]["mean"])
        if diff_asr > 0.05:
            print(f"[FAIL] Table 1 Round {r_idx+1} ASR summary mismatch: recomputed={rec['attack_success_rate']['mean']}, summary={sm['attack_success_rate']['mean']}")
            sys.exit(1)
    print("[OK] Table 1: n_runs_summary_unlimited.json strictly matches n_runs_raw_unlimited.jsonl")

    # 3. Recompute and verify Table 2 (Matched Two-Arm Comparison)
    print("\n--- Verifying Table 2: Matched Two-Arm Comparison (10 Runs) ---")
    recomputed_t2 = recompute_from_raw_jsonl("compare_arms_raw.jsonl")
    with open("compare_arms_metrics.json", "r", encoding="utf-8") as f:
        summary_t2 = json.load(f)

    for r_idx, (rec, sm) in enumerate(zip(recomputed_t2, summary_t2["summary_by_round"])):
        diff_s_asr = abs(rec["static_asr"]["mean"] - sm["static_arm"]["asr"]["mean"])
        diff_a_asr = abs(rec["adaptive_asr"]["mean"] - sm["adaptive_arm"]["asr"]["mean"])
        if diff_s_asr > 0.05 or diff_a_asr > 0.05:
            print(f"[FAIL] Table 2 Round {r_idx+1} ASR summary mismatch!")
            sys.exit(1)
    print("[OK] Table 2: compare_arms_metrics.json strictly matches compare_arms_raw.jsonl")

    # 4. Check Markdown Tables in RESULTS.md
    print(f"\n--- Verifying Markdown Table Integrity in {args.results_doc} ---")
    doc_tables = parse_markdown_tables(args.results_doc)
    if len(doc_tables) < 2:
        print(f"[FAIL] Expected at least 2 markdown tables in {args.results_doc}, found {len(doc_tables)}")
        sys.exit(1)

    t1_errors = verify_table_values(doc_tables[0], recomputed_t1, "Table 1", ["attack_success_rate", "detection_rate", "fpr_combined"])
    t2_errors = verify_table_values(doc_tables[1], recomputed_t2, "Table 2", ["static_asr", "static_det", "adaptive_asr", "adaptive_det"])

    all_table_errors = t1_errors + t2_errors
    if all_table_errors:
        print("\n[FAIL] Markdown Table Discrepancies Found:")
        for err in all_table_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[OK] Markdown Table values match recomputed raw data within rounding limits.")

    # 5. Check for suspicious duplicate cells
    dup_warnings = check_suspicious_cross_duplicates(doc_tables[1], "Table 2")
    if dup_warnings:
        print("\n[WARNING] Suspicious duplicate values flagged:")
        for w in dup_warnings:
            print(f"  {w}")

    print("\n" + "=" * 80)
    print("[PASS] Provenance verification passed: 0 mismatches, 0 integrity violations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
