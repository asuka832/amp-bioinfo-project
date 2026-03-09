import os
import sys
import time

import pandas as pd
import requests
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_project_root() -> str:
    current_script = os.path.abspath(__file__)
    pipeline_dir = os.path.dirname(current_script)
    return os.path.dirname(os.path.dirname(os.path.dirname(pipeline_dir)))


def fetch_structure(sequence: str, save_path: str) -> bool:
    # 调用 ESMFold API，成功时保存 PDB 文件
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    try:
        response = requests.post(url, data=sequence, verify=False, timeout=60)
        if response.status_code == 200:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return True
        print(f"请求失败，状态码: {response.status_code}")
        return False
    except Exception as e:
        print(f"网络错误: {e}")
        return False


def select_safe_top10(df: pd.DataFrame) -> pd.DataFrame:
    # 只保留安全序列并取前 10 条
    safe_candidates = df[df["Safety_Status"].astype(str).str.contains("Safe", na=False)].copy()
    if safe_candidates.empty:
        return safe_candidates

    if "Moment" in safe_candidates.columns:
        safe_candidates = safe_candidates.sort_values(by="Moment", ascending=False)
    elif "Score" in safe_candidates.columns:
        safe_candidates = safe_candidates.sort_values(by="Score", ascending=False)

    return safe_candidates.head(10)


def main():
    project_root = get_project_root()
    input_file = os.path.join(
        project_root,
        "data",
        "processed",
        "final_candidates",
        "Final_Perfect_Candidates.csv",
    )
    output_folder = os.path.join(project_root, "data", "processed", "structures")
    os.makedirs(output_folder, exist_ok=True)

    print("[Step 4] 开始结构预测")

    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        sys.exit(1)

    if "Safety_Status" not in df.columns:
        print("错误: 缺少 Safety_Status 列，请先完成 Step 3")
        sys.exit(1)

    target_list = select_safe_top10(df)
    print(f"可用于结构预测的安全序列数: {len(target_list)}")

    if target_list.empty:
        print("没有可用的安全序列，结束")
        sys.exit(0)

    success_count = 0
    for idx, row in enumerate(target_list.itertuples(index=False), start=1):
        seq = str(row.Sequence).strip()
        filename = f"Safe_Rank_{idx}_{seq[:5]}.pdb"
        save_path = os.path.join(output_folder, filename)

        if os.path.exists(save_path):
            print(f"[{idx}/{len(target_list)}] 已存在，跳过: {filename}")
            success_count += 1
            continue

        print(f"[{idx}/{len(target_list)}] 预测中: {seq[:10]}...")
        if fetch_structure(seq, save_path):
            print(f"保存成功: {filename}")
            success_count += 1
        else:
            print(f"预测失败: {filename}")

        time.sleep(1.5)

    print(f"Step 4 完成，成功获得 {success_count} 个结构文件")
    print(f"输出目录: {output_folder}")


if __name__ == "__main__":
    main()