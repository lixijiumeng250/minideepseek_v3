# MiniDeepSeek

个人复现mini DeepSeek-V3 预训练项目，支持多模态数据处理和模型训练。

## 快速开始

### 1. 数据集下载

使用 HuggingFace 镜像站下载数据集：

```bash
export HF_ENDPOINT=https://hf-mirror.com
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
sudo apt install aria2 git-lfs
git lfs install

# 下载数据集
./hfd.sh Skywork/SkyPile-150B --dataset --tool aria2c -x 10 --local-dir /path/to/data/SkyPile
```

运行数据预处理：
- `step1_data/step1_dataclean.py` - 数据去重和预处理
- `step1_data/split_jsonl.py` - 数据分割
- `step1_data/PdfProcess.py` - PDF 解析（多模态数据）

### 2. 数据清洗

使用 datajuicer 进行数据清洗：

```bash
git clone https://github.com/datajuicer/data-juicer.git
cd data-juicer
pip install -e .

# 运行清洗脚本
bash step2_clean/run_step2_skypile.sh
```

### 3. Tokenizer 处理

对清洗后的数据进行 tokenizer 和打包：

- `tokenizer/datapacked.py` - tokenizer 处理和打包，生成 .bin 文件
- `tokenizer/dataset.py` - 数据读取定义

### 4. 模型训练

安装依赖：

```bash
pip install accelerate deepspeed torch
```

配置 accelerate：

```bash
accelerate config
```

开始训练：

```bash
accelerate launch --num_processes=2 pretrain.py \
    --use_wandb \
    --wandb_project "minideepseek" \
    --data_path /path/to/your/data \
    --epochs 1 \
    --micro_batch_size 4 \
    --learning_rate 1e-4
```

## 主要参数

- `--data_path`: 训练数据路径（包含 .bin 文件）
- `--epochs`: 训练轮数
- `--micro_batch_size`: 每个 GPU 的批次大小
- `--accumulation_steps`: 梯度累积步数（默认 6）
- `--learning_rate`: 学习率（默认 1e-4）
- `--lambda_mtp`: MTP loss 权重（默认 0.5）
- `--use_wandb`: 启用 Weights & Biases 日志

## 项目结构

```
minideepseek/
├── model/              # 模型定义
├── tokenizer/          # Tokenizer 和数据加载
├── step1_data/         # 数据预处理
├── step2_clean/        # 数据清洗配置
├── pretrain.py         # 训练主脚本
└── zero1_config.json   # DeepSpeed ZeRO-1 配置
```

