import subprocess
import sys
import time


target_prefixes = ["G", "K", "R", "W", "L"]
target_lengths = range(13, 19)
samples_per_batch = 30
device = "0"


def get_project_root() -> str:
    current_script = os.path.abspath(__file__)
    pipeline_dir = os.path.dirname(current_script)
    return os.path.dirname(os.path.dirname(pipeline_dir))


def run_generation_task(generator_script, model_path, save_path, prefix, length):
    # AMPGenix 需要区间格式，例如 14-14
    length_str = f"{length}-{length}"
    cmd_args = [
        sys.executable,
        generator_script,
        "--model_path",
        model_path,
        "--save_samples_path",
        save_path,
        "--device",
        device,
        "--nsamples",
        str(samples_per_batch),
        "--ntokens",
        length_str,
        "--prefix",
        prefix,
        "--save_samples",
    ]
    subprocess.run(cmd_args, check=True)


def main():
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models", "AMPGenix")
    save_path = os.path.join(project_root, "data", "01_raw")
    generator_script = os.path.join(project_root, "src", "original", "AMPGenix.py")

    os.makedirs(save_path, exist_ok=True)

    total_tasks = len(target_prefixes) * len(list(target_lengths))
    print("[Step 1] 开始生成候选序列")
    print(f"预计总产量: {total_tasks * samples_per_batch} 条")

    start_time = time.time()
    task_count = 0

    for prefix in target_prefixes:
        for length in target_lengths:
            task_count += 1
            print(f"[{task_count}/{total_tasks}] 生成中: 前缀={prefix}, 长度={length}")
            try:
                run_generation_task(generator_script, model_path, save_path, prefix, length)
                print(f"批次完成: 前缀={prefix}, 长度={length}")
            except subprocess.CalledProcessError as e:
                print(f"批次失败: 前缀={prefix}, 长度={length}, 错误码={e.returncode}")
            except Exception as e:
                print(f"批次异常: 前缀={prefix}, 长度={length}, 错误={e}")
            time.sleep(2)

    duration = (time.time() - start_time) / 60
    print(f"Step 1 完成，总耗时: {duration:.1f} 分钟")


if __name__ == "__main__":
    main()