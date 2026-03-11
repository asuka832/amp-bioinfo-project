# Pipeline 目录与数据流说明

## 主流程（可一键运行）

```bash
python main.py
```

执行顺序：
1. `scripts/pipeline/step01_generate.py`
2. `scripts/pipeline/step02_score_amp.py`
3. `scripts/pipeline/step03_filter_toxicity.py`
4. `scripts/pipeline/step04_fold_structure.py`

## 拓展流程（系统发育与 iTOL）

```bash
python scripts/pipeline/step05_clean_blast.py
python scripts/pipeline/step06_make_itol.py
```

## 数据目录规范

- `data/raw/blast/raw_blast.txt`：BLAST 原始文本输入
- `data/interim/01_generated/`：Step01 生成序列
- `data/interim/02_amp_scores/Total_Scores.csv`：Step02 活性打分结果
- `data/processed/final_candidates/Final_Perfect_Candidates.csv`：Step03 最终候选
- `data/processed/structures/`：Step04 PDB 结构
- `data/processed/phylo/family_tree.fasta`：Step05 输出 FASTA
- `data/processed/itol/`：Step06 iTOL 注释文件
