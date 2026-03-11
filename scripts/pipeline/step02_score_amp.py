import glob
import os
import subprocess
import sys
import time

import pandas as pd


device = "0"


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def extract_sequences(input_folder: str) -> list:
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    all_sequences = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, header=None)
            seqs = df.iloc[:, -1].tolist()
            for seq in seqs:
                clean_seq = "".join(ch for ch in str(seq) if ch.isalpha())
                if len(clean_seq) > 5:
                    all_sequences.append(clean_seq)
        except Exception:
            continue

    return list(set(all_sequences))


def write_temp_input(temp_input: str, sequences: list):
    with open(temp_input, "w", encoding="utf-8") as f:
        f.write("Sequence\n")
        for seq in sequences:
            f.write(f"{seq}\n")


def run_ampsorter(predictor_script, model_path, weight_file, temp_input, output_file):
    cmd_args = [
        sys.executable,
        predictor_script,
        "--model_path", model_path,
        "--classifier_path", weight_file,
        "--candidate_amp_path", temp_input,
        "--raw_data_path", temp_input,
        "--output_path", output_file,
        "--device", device,
    ]
    subprocess.run(cmd_args, check=True)


def main():
    project_root = get_project_root()

    input_folder = os.path.join(project_root, "data", "interim", "01_generated")
    output_folder = os.path.join(project_root, "data", "interim", "02_amp_scores")
    model_path = os.path.join(project_root, "models", "AMPSorter")
    weight_file = os.path.join(model_path, "pytorch_model.bin")
    predictor_script = os.path.join(project_root, "src", "original", "AMPSorter_predictor.py")
    output_file = os.path.join(output_folder, "Total_Scores.csv")
    temp_input = os.path.join(output_folder, "temp_merged_input.txt")

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(predictor_script):
        print(f"错误: 找不到脚本文件 {predictor_script}")
        sys.exit(1)

    print("[Step 2] 开始读取并整理序列")
    sequences = extract_sequences(input_folder)
    print(f"待打分序列数: {len(sequences)}")

    if not sequences:
        print("错误: 没有提取到有效序列")
        sys.exit(1)

    write_temp_input(temp_input, sequences)

    print("开始调用 AMPSorter 打分")
    start_time = time.time()

    try:
        run_ampsorter(predictor_script, model_path, weight_file, temp_input, output_file)
        print(f"打分完成，结果保存在: {output_file}")
        if os.path.exists(temp_input):
            os.remove(temp_input)
    except subprocess.CalledProcessError as e:
        print(f"打分失败，错误码: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("错误: 找不到 Python 解释器或目标脚本")
        sys.exit(1)

    duration = time.time() - start_time
    print(f"Step 2 完成，耗时: {duration:.1f} 秒")


if __name__ == "__main__":
    main()
