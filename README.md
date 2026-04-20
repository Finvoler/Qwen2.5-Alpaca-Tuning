# Qwen2.5-Alpaca-Tuning

> **Instruction Fine-Tuning of Qwen2.5-0.5B on the Alpaca Dataset with Custom Decoding Strategies**

This project implements full-parameter supervised fine-tuning (SFT) of the [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) base language model on the [Alpaca-Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) instruction dataset. The goal is to transform a raw pre-trained model into a practical instruction-following assistant capable of natural dialogue.

---

## Highlights

- **Full-parameter SFT** — all model weights are updated during fine-tuning (no LoRA / adapter tricks)
- **Custom training loop** — manually implemented forward pass, loss masking, and optimizer step with PyTorch
- **Three decoding strategies** — greedy decoding, sampling (top-k / top-p / temperature), and beam search, all implemented from scratch
- **Batch training & batch inference** — efficient batched collation with dynamic padding and multi-query inference support

---

## Project Structure

```
.
├── train_inference.ipynb      # Main Kaggle notebook (training + inference in one place)
├── main/
│   ├── finetune.py            # Standalone training script
│   ├── inference.py           # Standalone inference & generation script
│   ├── plot.py                # Training loss visualization
│   └── README.md              # Module-level documentation
└── alpaca-cleaned/
    ├── alpaca_data_cleaned.json   # Instruction-tuning dataset (~52K samples)
    └── README.md
```

---

## Background

Modern large language models go through multiple training stages:

1. **Pre-training** — learns general language structure from massive corpora via next-token prediction
2. **Instruction fine-tuning (SFT)** — trains the model to follow human instructions using curated `(instruction, output)` pairs
3. **RLHF** *(out of scope here)* — further aligns model outputs with human preferences

This project focuses on **stage 2**: taking the Qwen2.5-0.5B base model (which can predict the next token but cannot hold a conversation) and turning it into a model that understands and responds to user instructions.

---

## Dataset

We use the **Alpaca-Cleaned** dataset — a quality-filtered version of Stanford's original Alpaca dataset with ~52K English instruction-following samples. Each sample contains:

| Field | Description |
|---|---|
| `instruction` | The task or question posed to the model |
| `input` | Optional context (empty string if not applicable) |
| `output` | The expected high-quality response |

Samples with combined token length > 512 are filtered out to keep training efficient.

---

## Tokenization & Label Masking

Chat-format texts are constructed with `tokenizer.apply_chat_template`. To ensure the model only learns to generate the **assistant's response** (and not to "predict" the instruction itself), we mask all prompt tokens in `labels` with `-100`:

```
<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n  ← labels = -100 here
{output}<|im_end|>                                                   ← labels = token ids here
```

The EOS token is set to `<|im_end|>` to be consistent with Qwen's chat format.

---

## Training

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` |
| Optimizer | AdamW |
| Learning rate | `5e-5` |
| Batch size | 32 |
| Max sequence length | 512 tokens |
| Epochs | 3 |
| Hardware | Kaggle GPU (T4 x2 or P100) |

The training loop is implemented manually in PyTorch without using HuggingFace `Trainer`. At each step:
1. Batch inputs are forwarded through the model
2. Logits are shifted by one position for next-token prediction
3. Cross-entropy loss is computed (ignoring `-100` labels)
4. Gradients are backpropagated and the optimizer updates all parameters

Checkpoints are saved after each epoch to `./ckpts/checkpoint-epoch-{n}/`.

---

## Decoding Strategies

All three decoding strategies are implemented from scratch in `inference.py` (and in `train_inference.ipynb`):

### 1. Greedy Decoding
At each step, the token with the highest probability is selected deterministically.  
Fast and consistent, but can produce repetitive or bland responses.

### 2. Sampling Decoding
Tokens are sampled from the distribution, controlled by:
- **Temperature** — sharpens (< 1) or flattens (> 1) the distribution
- **Top-k** — restricts sampling to the top *k* tokens
- **Top-p (nucleus sampling)** — restricts sampling to the smallest set of tokens whose cumulative probability exceeds *p*

Produces more diverse and creative responses at the cost of consistency.

### 3. Beam Search *(Bonus)*
Maintains `num_beams` candidate sequences simultaneously and selects the one with the highest cumulative log-probability, optionally adjusted by a `length_penalty`.

---

## Quick Start

### Dependencies

```bash
pip install -r requirements.txt
```

### Fine-Tuning

```bash
python main/finetune.py
```

Model checkpoints will be saved under `./ckpts/`.

### Inference

```bash
python main/inference.py
```

Loads `./ckpts/checkpoint-epoch-3` and runs all three decoding strategies on sample queries.

### Training Curve

```bash
python main/plot.py
```

Reads `training_loss.json` produced during training and saves `loss_curve.png`.

---

## Results & Observations

**Before fine-tuning**, the base model responds to `"Give me a brief introduction to Shanghai Jiao Tong University."` with incoherent continuation text — it simply continues the token sequence without understanding the instruction format.

**After 3 epochs of SFT**, the model correctly identifies the instruction, structures a reply, and generates topically relevant content. For example:

```
Q: Give me a brief introduction to Shanghai Jiao Tong University.
A: Shanghai Jiao Tong University (SJTU) is a leading research university
   located in Shanghai, China. Founded in 1896, SJTU is one of the oldest
   and most prestigious universities in China...
```

**Sampling vs. greedy**: Sampling with `temperature=0.7, top_p=0.9` produces more varied and sometimes more natural-sounding answers for open-ended queries (e.g., "Tell me a joke"), while greedy decoding tends to be more focused for factual questions.

**Beam search**: With `num_beams=4`, mathematical reasoning queries benefit from the wider search space — the model is more likely to produce a coherent step-by-step solution.

---

## Pretrained Model Weights

The fine-tuned model (`checkpoint-epoch-3`) is available for download:

- **SJTU Cloud Drive**: [Download Link](https://pan.sjtu.edu.cn/web/share/11a1c1d12ba6a73d5392e6ad6f8087f4)

---

## License

- Code: [MIT License](LICENSE)
- Dataset (Alpaca-Cleaned): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Base Model (Qwen2.5-0.5B): [Qwen License](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/LICENSE)
