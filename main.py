import os
import time
import sys
import subprocess


def run_step(script_name, description):
    print(f"\n{'=' * 60}")
    print(f"正在执行步骤: {description}")
    print(f"脚本路径: scripts/pipeline/{script_name}")
    print(f"{'=' * 60}\n")

    script_path = os.path.join("scripts", "pipeline", script_name)

    try:
        subprocess.run([sys.executable, script_path], check=True)
        exit_code = 0
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode

    if exit_code != 0:
        print(f"错误：{script_name} 执行失败！流程终止。")
        sys.exit(1)
    else:
        print(f"完成：{description}")
        time.sleep(2)


if __name__ == "__main__":
    print("AMP 候选肽挖掘主流程启动")

    # 1. 生成序列
    run_step("step01_generate.py", "AMPGenix 生成候选序列")

    # 2. 预测活性
    run_step("step02_score_amp.py", "AMPSorter 活性打分")

   

    print("\n全流程执行完毕！请查看 data/processed/final_candidates")