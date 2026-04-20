import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "./dataset/train.csv"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
# print(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

dataset = datasets.Dataset.from_csv(DATASET_PATH)
# for sample in dataset.select(range(10)):    # 查看前10个样本。思考应该怎么将样本组织成单条完整文本？
#     print(sample)

tokenizer.apply_chat_template([
    {"role": "user", "content": "This is a question."},
    {"role": "assistant", "content": "I'm the answer!"}
], tokenize=False
)

# print(tokenizer.eos_token)  # 原来的终止符
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id  # 也要修改模型的终止符


# messages = [
#     {"role": "user", "content": "Give me a brief introduction to Shanghai Jiao Tong University."},
# ]
# text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# with torch.no_grad():
#     lm_inputs_src = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)
#     generate_ids = model.generate(**lm_inputs_src, max_new_tokens=150, do_sample=False)
# pred_str = tokenizer.decode(generate_ids[0][lm_inputs_src.input_ids.size(1):], skip_special_tokens=True)
# print(pred_str)


# --- Process the Dataset ---
import copy
def tokenize_function(sample):
    input_ids = None
    labels = None
    # TODO: 完成函数，使之能够对数据集中的每个样本进行正确的tokenize，生成训练时用于输入模型的input_ids和labels。
    # 思考一下，labels的标签应该如何设置，才能让模型只对output的部分进行计算loss？
    
    messages = [
        {"role": "user", "content": sample['instruction']},
        {"role": "assistant", "content": sample['output']}
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer(full_text)['input_ids']

    user_message = [
        {"role": "user", "content": sample['instruction']}
    ]
    user_text = tokenizer.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    user_input_ids = tokenizer(user_text)['input_ids']

    labels = list(input_ids)
    user_len = len(user_input_ids)
    labels[: user_len] = [-100] * user_len

    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(
    tokenize_function, remove_columns=dataset.column_names
).filter(
    lambda x: len(x["input_ids"]) <= 512
)

from torch.utils.data import DataLoader

def collate_fn(batch):
    input_ids = None
    attention_mask = None
    labels = None
    # TODO: 完成函数，使之能够对取出的batch样本进行处理，生成适合模型输入的input_ids, attention_mask和labels
    input_ids_list = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    labels_list = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]

    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    labels = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100
    )

    attention_mask = (input_ids != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)).long()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 根据显存占用情况，可以适当调整batch_size
train_dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


# --- Training ---
from tqdm import tqdm

step = 0
# TODO: 定义你的优化器与损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

model.train()
for epoch in range(3):
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        loss = None
        # TODO: 手动完成单步的训练步骤
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        loss = loss_fn(flat_logits, flat_labels)

        loss.backward()
        optimizer.step()
        
        step += 1
        if step % 100 == 0:
            print(f"Step {step}\t| Loss: {loss.item()}")
    model.save_pretrained(f"./ckpts/checkpoint-epoch-{epoch + 1}")
    tokenizer.save_pretrained(f"./ckpts/checkpoint-epoch-{epoch + 1}")


# --- Testing Model ---

sft_model = AutoModelForCausalLM.from_pretrained("./ckpts/checkpoint-epoch-3", device_map="auto", torch_dtype="auto")
messages = [
    {"role": "user", "content": "Give me a brief introduction to Shanghai Jiao Tong University."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
with torch.no_grad():
    lm_inputs_src = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(sft_model.device)
    generate_ids = sft_model.generate(**lm_inputs_src, max_new_tokens=150, do_sample=False)
pred_str = tokenizer.decode(generate_ids[0][lm_inputs_src.input_ids.size(1):], skip_special_tokens=True)
print(pred_str)