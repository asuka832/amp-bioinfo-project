import math
import os
import subprocess
import sys

import pandas as pd


device = "0"
THRESHOLD_AMPSORTER = 0.9
THRESHOLD_CHARGE = 3
THRESHOLD_MOMENT = 0.3

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


def calculate_physicochem(seq):
    # 计算长度、电荷、疏水比例和疏水矩
    seq = "".join(ch for ch in str(seq).upper() if ch.isalpha())
    length = len(seq)
    if length == 0:
        return None

    charge = seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E")
    hydro_aa = ["A", "I", "L", "F", "W", "V", "M"]
    hydrophobicity = sum(seq.count(aa) for aa in hydro_aa) / length

    sum_sin, sum_cos = 0.0, 0.0
    for i, aa in enumerate(seq):
        h = hydrophobicity_scale.get(aa, 0)
        angle = math.radians(100 * i)
        sum_sin += h * math.sin(angle)
        sum_cos += h * math.cos(angle)
    moment = math.sqrt(sum_sin**2 + sum_cos**2) / length

    return pd.Series(
        [length, charge, hydrophobicity, moment],
        index=["Length", "Charge", "Hydrophobicity", "Moment"],
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 统一列名，确保有 Sequence 和 Score
    if "Sequence" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Sequence"})

    if "Score" not in df.columns:
        for name in ["prob_1", "probability", "prob", "pred", "score"]:
            if name in df.columns:
                df = df.rename(columns={name: "Score"})
                break

    if "Score" not in df.columns:
        last_col = df.columns[-1]
        if last_col != "Sequence":
            df = df.rename(columns={last_col: "Score"})

    if "Sequence" not in df.columns or "Score" not in df.columns:
        raise ValueError(f"无法识别必要列，当前列名: {list(df.columns)}")

    return df


def run_toxicity_predict(tox_script, tox_model, tox_weight, input_path, output_path):
    cmd_args = [
        sys.executable,
        tox_script,
        "--model_path",
        tox_model,
        "--classifier_path",
        tox_weight,
        "--candidate_pep_path",
        input_path,
        "--raw_data_path",
        input_path,
        "--output_path",
        output_path,
        "--device",
        device,
    ]
    subprocess.run(cmd_args, check=True)


def main():
    project_root = get_project_root()

    input_file = os.path.join(project_root, "data", "interim", "02_amp_scores", "Total_Scores.csv")
    output_dir = os.path.join(project_root, "data", "processed", "final_candidates")
    output_file = os.path.join(output_dir, "Final_Perfect_Candidates.csv")

    tox_model_path = os.path.join(project_root, "models", "BioToxiPept")
    tox_weight_file = os.path.join(tox_model_path, "pytorch_model.bin")
    tox_script_path = os.path.join(project_root, "src", "original", "BioToxiPept.py")

    temp_tox_input = os.path.join(output_dir, "temp_tox_input.txt")
    temp_tox_output = os.path.join(output_dir, "temp_tox_output.csv")

    os.makedirs(output_dir, exist_ok=True)

    print("[Step 3] 读取打分结果并做理化筛选")
    try:
        df = pd.read_csv(input_file)
        df = normalize_columns(df)
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        sys.exit(1)

    screened = df[df["Score"] > THRESHOLD_AMPSORTER].copy()
    print(f"分数筛选后剩余: {len(screened)}")
    if screened.empty:
        print("没有高分序列，流程结束")
        sys.exit(0)

    props = screened["Sequence"].apply(calculate_physicochem)
    combined = pd.concat([screened, props], axis=1)
    candidates = combined[
        (combined["Charge"] >= THRESHOLD_CHARGE)
        & (combined["Moment"] > THRESHOLD_MOMENT)
    ].copy()

    candidates = candidates.sort_values(by="Moment", ascending=False)
    print(f"理化筛选后剩余: {len(candidates)}")
    if candidates.empty:
        print("没有通过理化筛选的序列")
        sys.exit(0)

    if len(candidates) > 50:
        candidates = candidates.head(50)
        print("候选序列过多，只保留 Top 50 进行毒性预测")

    with open(temp_tox_input, "w", encoding="utf-8") as f:
        f.write("Sequence\n")
        for seq in candidates["Sequence"]:
            f.write(f"{seq}\n")

    print("开始运行毒性预测")
    try:
        run_toxicity_predict(
            tox_script_path,
            tox_model_path,
            tox_weight_file,
            temp_tox_input,
            temp_tox_output,
        )
    except subprocess.CalledProcessError:
        print("毒性模型运行失败")
        if os.path.exists(temp_tox_input):
            os.remove(temp_tox_input)
        sys.exit(1)

    if not os.path.exists(temp_tox_output):
        print("毒性输出文件不存在")
        if os.path.exists(temp_tox_input):
            os.remove(temp_tox_input)
        sys.exit(1)

    try:
        tox_res = pd.read_csv(temp_tox_output)
        tox_scores = tox_res.iloc[:, -1].values
        if len(tox_scores) != len(candidates):
            print("毒性结果行数与候选序列不一致")
            sys.exit(1)

        candidates["Toxicity_Prob"] = tox_scores
        candidates["Safety_Status"] = candidates["Toxicity_Prob"].apply(
            lambda x: "Safe" if x < 0.5 else "Toxic"
        )
        candidates.to_csv(output_file, index=False)

        safe_count = (candidates["Safety_Status"] == "Safe").sum()
        print(f"Step 3 完成，安全序列数量: {safe_count}")
        print(f"结果文件: {output_file}")
    finally:
        if os.path.exists(temp_tox_input):
            os.remove(temp_tox_input)
        if os.path.exists(temp_tox_output):
            os.remove(temp_tox_output)


if __name__ == "__main__":
    main()