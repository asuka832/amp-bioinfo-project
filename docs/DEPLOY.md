# 部署指南（Windows + Conda）

## 1）克隆并进入项目目录

```bat
git clone https://github.com/asuka832/amp-bioinfo-project.git
cd Amp_project
```

## 2）一键配置环境

```bat
setup.bat
```

该脚本会自动完成：
- 根据 `environment.yml` 创建或更新 `amp_env`
- 验证 `pandas` 是否可导入
- 验证 `torch` 是否可导入并检查 CUDA 可用性

## 3）运行主流程

```bat
run.bat
```

`run.bat` 会在当前进程临时设置 `NO_PROXY=*`，用于规避系统代理异常导致 `transformers` 下载 tokenizer 失败的问题。

## 4）VS Code 解释器设置

请设置为：

`C:\Users\<your_user>\.conda\envs\amp_env\python.exe`

也可在终端执行以下命令快速自检：

```bat
conda activate amp_env
python -c "import pandas, torch; print(pandas.__version__, torch.__version__)"
```

## 说明

- 首次运行可能会从 Hugging Face 下载 tokenizer 相关文件。
- 如果网络受限，建议提前准备 tokenizer 文件，并将代码改为本地加载模式。
