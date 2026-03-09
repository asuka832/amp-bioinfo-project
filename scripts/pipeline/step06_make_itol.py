import math
import os
import sys

import pandas as pd


hydrophobicity_scale = {
    "A": 0.62,
    "R": -2.53,
    "N": -0.78,
    "D": -0.90,
    "C": 0.29,
    "Q": -0.85,
    "E": -0.74,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "L": 1.06,
    "K": -1.50,
    "M": 0.64,
    "F": 1.19,
    "P": 0.12,
    "S": -0.18,
    "T": -0.05,
    "W": 0.81,
    "Y": 0.26,
    "V": 1.08,
}


def get_project_root() -> str:
    current_script = os.path.abspath(__file__)
    pipeline_dir = os.path.dirname(current_script)
    return os.path.dirname(os.path.dirname(os.path.dirname(pipeline_dir)))


def calculate_props(seq: str):
    # 计算电荷和疏水矩
    clean_seq = "".join(ch for ch in str(seq).upper() if ch.isalpha())
    length = len(clean_seq)
    if length == 0:
        return 0, 0

    charge = clean_seq.count("K") + clean_seq.count("R") - clean_seq.count("D") - clean_seq.count("E")

    sum_sin, sum_cos = 0.0, 0.0
    for i, aa in enumerate(clean_seq):
        h = hydrophobicity_scale.get(aa, 0)
        angle = math.radians(100 * i)
        sum_sin += h * math.sin(angle)
        sum_cos += h * math.cos(angle)

    moment = math.sqrt(sum_sin**2 + sum_cos**2) / length
    return charge, moment


def load_safety_dict(final_report_csv: str) -> dict:
    # 建立序列前10位到安全状态的映射
    safety_dict = {}
    if not os.path.exists(final_report_csv):
        return safety_dict

    try:
        df = pd.read_csv(final_report_csv)
        for _, row in df.iterrows():
            if "Sequence" in row and "Safety_Status" in row:
                seq_key = str(row["Sequence"]).strip()[:10]
                safety_dict[seq_key] = str(row["Safety_Status"])
    except Exception:
        pass

    return safety_dict


def write_headers(paths: dict):
    with open(paths["source"], "w", encoding="utf-8") as f:
        f.write(
            "DATASET_COLORSTRIP\nSEPARATOR TAB\nDATASET_LABEL\tSource (Red=AI)\n"
            "COLOR\t#ff00ff\nSTRIP_WIDTH\t25\nDATA\n"
        )
    with open(paths["charge"], "w", encoding="utf-8") as f:
        f.write(
            "DATASET_SIMPLEBAR\nSEPARATOR TAB\nDATASET_LABEL\tCharge (+)\n"
            "COLOR\t#ff0000\nWIDTH\t100\nDATA\n"
        )
    with open(paths["moment"], "w", encoding="utf-8") as f:
        f.write(
            "DATASET_HEATMAP\nSEPARATOR TAB\nDATASET_LABEL\tHydrophobic Moment\n"
            "COLOR\t#0000ff\nFIELD_LABELS\tMoment\nCOLOR_MIN\t#ffffff\n"
            "COLOR_MAX\t#0000ff\nDATA\n"
        )
    with open(paths["safety"], "w", encoding="utf-8") as f:
        f.write(
            "DATASET_COLORSTRIP\nSEPARATOR TAB\nDATASET_LABEL\tSafety (Green=Safe)\n"
            "COLOR\t#00ff00\nSTRIP_WIDTH\t25\nDATA\n"
        )
    with open(paths["info"], "w", encoding="utf-8") as f:
        f.write("DATASET_TEXT\nSEPARATOR TAB\nDATASET_LABEL\tValues\nCOLOR\t#000000\nDATA\n")


def append_line(filepath: str, line: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    project_root = get_project_root()

    input_fasta = os.path.join(project_root, "data", "processed", "phylo", "family_tree.fasta")
    final_report_csv = os.path.join(
        project_root, "data", "processed", "final_candidates", "Final_Perfect_Candidates.csv"
    )
    output_dir = os.path.join(project_root, "data", "processed", "itol")
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "source": os.path.join(output_dir, "itol_01_source_strip.txt"),
        "charge": os.path.join(output_dir, "itol_02_charge_bar.txt"),
        "moment": os.path.join(output_dir, "itol_03_moment_heatmap.txt"),
        "safety": os.path.join(output_dir, "itol_04_safety_strip.txt"),
        "info": os.path.join(output_dir, "itol_05_info_text.txt"),
    }

    if not os.path.exists(input_fasta):
        print(f"错误: 找不到文件 {input_fasta}")
        sys.exit(1)

    safety_dict = load_safety_dict(final_report_csv)
    write_headers(files)

    count = 0
    current_id = ""

    with open(input_fasta, "r", encoding="utf-8") as f_in:
        for raw_line in f_in:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:]
                continue

            seq = line
            charge, moment = calculate_props(seq)
            is_ai = "My_AI" in current_id

            color_src = "#ff0000" if is_ai else "#cccccc"
            append_line(files["source"], f"{current_id}\t{color_src}\n")
            append_line(files["charge"], f"{current_id}\t{charge}\n")
            append_line(files["moment"], f"{current_id}\t{moment:.3f}\n")

            color_safe = "#cccccc"
            if is_ai:
                status = safety_dict.get(seq[:10], "Unknown")
                if "Safe" in status:
                    color_safe = "#00ff00"
                elif "Toxic" in status:
                    color_safe = "#ff0000"
            append_line(files["safety"], f"{current_id}\t{color_safe}\n")

            if is_ai:
                info = f"Q={charge}, uH={moment:.2f}"
                append_line(files["info"], f"{current_id}\t{info}\t1\t#000000\tbold\t1.5\t0\n")

            count += 1

    print(f"Step 6 完成，生成 {count} 条 iTOL 数据")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()