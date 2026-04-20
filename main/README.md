# NLP 大作业一：预训练模型的指令微调及对话文本生成

## 简介

本此作业基于 Qwen2.5-0.5B 基底模型，利用 Alpaca 指令数据集进行全量指令微调（SFT），并实现一个支持多种解码策略（贪心解码、采样解码、束搜索解码）的对话模型。

## 文件说明

- `finetune.py`: 包含数据预处理、模型加载、训练循环及模型保存的代码。实现了对 Alpaca 数据集的指令微调。
- `inference.py`: 包含模型加载及文本生成的代码。实现了贪心解码、采样解码（Top-k, Top-p, Temperature）以及束搜索解码（Beam Search）。
- `plot.py`：对 `fintune.py` 生成的 `training_loss.json` 进行可视化。
- `dataset/`: 存放训练数据的目录。
- `ckpts/`: 存放训练过程中及最终保存的模型权重。

## 运行指南

### 1. 模型微调

运行以下命令开始训练模型：

```bash
python finetune.py
```

训练过程中，模型将运行 3 个 epoch，每个 epoch 结束后会自动保存检查点到 `ckpts/` 目录下。

### 2. 对话生成

训练完成后，运行以下命令测试模型的对话生成能力：

```bash
python inference.py
```

该脚本将加载 `ckpts/checkpoint-epoch-3` 中的模型，并演示贪心解码、采样解码和束搜索解码的效果。

### 3. 可视化

完成指令微调后即可对训练损失进行可视化：

```bash
python plot.py
```

将自动加载 `training_loss.json`。

## 模型下载

训练好的模型权重已上传至 [交大云盘](https://pan.sjtu.edu.cn/web/share/11a1c1d12ba6a73d5392e6ad6f8087f4)。

## 小组信息

- 成员 1: 叶墨涵 523030910171
- 成员 2: 邓建斌 523030910177
- 成员 3: 陈祺 523030910169