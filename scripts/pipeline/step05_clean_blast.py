import os
import sys


my_ai_seq = "WKLLKKLLKLLKKL"


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_blast_file(input_file: str):
    # 从 raw_blast.txt 中提取序列 ID 和序列
    valid_sequences = []
    current_id = None

    with open(input_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("AP") and len(line) < 15 and "Alignment" not in line:
                current_id = line
                continue

            if "-Alignment Result-:" in line:
                parts = line.split(":", 1)
                if len(parts) < 2:
                    continue

                clean_seq = "".join(ch for ch in parts[1].replace(" ", "").upper() if ch.isalpha())
                if len(clean_seq) <= 5:
                    continue

                seq_name = current_id if current_id else f"Natural_Pep_{len(valid_sequences) + 1}"
                valid_sequences.append((seq_name, clean_seq))
                current_id = None

    return valid_sequences


def write_fasta(output_file: str, ai_seq: str, natural_sequences: list):
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(f">My_AI_Peptide\n{ai_seq}\n")
        for name, seq in natural_sequences:
            f_out.write(f">{name}\n{seq}\n")


def main():
    project_root = get_project_root()
    input_file = os.path.join(project_root, "data", "raw", "blast", "raw_blast.txt")
    output_file = os.path.join(project_root, "data", "processed", "phylo", "family_tree.fasta")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)

    print("[Step 5] 开始清洗 BLAST 结果")
    valid_sequences = parse_blast_file(input_file)
    write_fasta(output_file, my_ai_seq, valid_sequences)

    print(f"提取完成: 1 条 AI + {len(valid_sequences)} 条天然序列")
    print(f"结果文件: {output_file}")

    if valid_sequences:
        first_name, first_seq = valid_sequences[0]
        print(f"示例: >{first_name} -> {first_seq[:10]}...")
    else:
        print("未提取到天然序列，请检查 raw_blast.txt 格式")


if __name__ == "__main__":
    main()
