import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from typing import Union, List

MODEL_PATH = "./ckpts/checkpoint-epoch-3"    # 你训练好的模型路径

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

def generate(
    model: AutoModelForCausalLM,
    query: Union[str, List[str]],
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    num_beams: int = 1,
    length_penalty: float = 1.0,
) -> Union[str, List[str]]:
    """
    使用模型model进行文本生成。
    Args:
        model: 用于生成的语言模型
        query: 用户输入的查询。可以是单个字符串，或者是一个字符串列表【附加2】
        max_new_tokens: 生成的最大新token数量
        do_sample: 是否使用采样生成文本。仅当为True时，后续的temperature、top_p、top_k参数才会生效
        temperature: 采样时的温度参数
        top_p: 采样时的top-p参数
        top_k: 采样时的top-k参数
        num_beams: 束搜索同时维护的束的数量。仅当`num_beams > 1`时，才会启用束搜索【附加3】
        length_penalty: 启用束搜索时的长度惩罚系数【附加3】
    Returns:
        生成的文本。如果输入是单个字符串，则返回单个字符串；如果输入是字符串列表，则返回字符串列表【附加2】
    """
    # TODO: 完成函数，实现文本生成
    # 统一批次
    if isinstance(query, str):
        query = [query]

    device = model.device

    model_inputs = tokenizer(query, return_tensors='pt', padding=True).to(device)
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = model.generation_config.eos_token_id
    generated_ids = None
    if num_beams > 1:
        # beam search
        generated_ids = _beam_search_step(
            model, input_ids, attention_mask, max_new_tokens, num_beams, length_penalty, pad_token_id, eos_token_id
        )
    else:
        # greedy or sampling
        generated_ids = _greedy_sample_step(
            model, input_ids, attention_mask, 
            max_new_tokens, do_sample, temperature, top_p, top_k, pad_token_id, eos_token_id
        )

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return output_text

def _greedy_sample_step(
    model, input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p, top_k, pad_token_id, eos_token_id
):
    """
    贪心解码或采样解码
    """
    batch_size, _ = input_ids.shape
    device = input_ids.device
    
    # if meet eos_token
    # [batch_size, 1]
    finished_sequences = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # [batch, vocab]
        next_token_logits = outputs.logits[:, -1, :]

        # sampling
        if do_sample:
            # temperature
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # top-k
            if top_k > 0:
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                min_top_k_logits = top_k_logits[:, -1].unsqueeze(-1)

                next_token_logits = torch.where(
                    next_token_logits < min_top_k_logits,
                    torch.tensor(float('-inf'), device=device),
                    next_token_logits
                )

            # yop-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            # sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1) # [batch, 1]
            next_tokens = next_tokens.squeeze(1) # [batch]

        # greedy
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1) # [batch]
        
        # if eos
        is_eos = (next_tokens == eos_token_id)
        
        finished_sequences = finished_sequences | is_eos.unsqueeze(1)

        # 如果当前样本已完成，将 next_token 替换为 pad_token_id (如果设置了的话)，避免生成乱码
        if pad_token_id is not None:
            # 只有那些已经标记为 finished 的样本才会被替换
            # 注意：如果是刚生成的 EOS，我们保留 EOS；如果是之前就 finished 的，替换为 PAD
            # 但为了简化，这里只要 finished 为 True，后续就都 append 结果（实际上生成函数最后 decode 会去掉 special tokens）
            # 更严谨的做法是：如果 finished_sequences 之前已经是 True，则 next_token = pad_token
            # 这里简单处理：不做特殊替换，依赖 batch_decode(skip_special_tokens=True)
            pass

        # update input_ids
        next_tokens = next_tokens.unsqueeze(-1) # [batch, 1]
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        # update attention mask
        new_mask = torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        if finished_sequences.all():
            break

    return input_ids

def _beam_search_step(
    model, input_ids, attention_mask, max_new_tokens, num_beams, length_penalty, pad_token_id, eos_token_id
):
    """
    束解码
    """
    batch_size, seq_len = input_ids.shape
    vocab_size = model.config.vocab_size
    device = input_ids.device

    # [batch, len] -> [batch * beams, len]
    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)


    # [batch, beams]
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # [batch * beams]

    cur_len = seq_len
    
    for _ in range(max_new_tokens):

        outputs = model(input_ids, attention_mask=attention_mask)

        # [batch * beams, vocab]
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        
        # [batch * beams, vocab]
        next_scores = beam_scores.unsqueeze(-1) + next_token_scores

        # [batch, beams * vocab]
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)
        
        # next_scores: [batch, beams], next_tokens: [batch, beams]
        next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1, largest=True, sorted=True)
        
        beam_indices = next_tokens // vocab_size
        token_indices = next_tokens % vocab_size

        beam_offset = torch.arange(batch_size, device=device) * num_beams
        global_beam_indices = beam_indices + beam_offset.unsqueeze(-1)
        global_beam_indices = global_beam_indices.view(-1) # Flatten
        
        input_ids = input_ids[global_beam_indices]
        attention_mask = attention_mask[global_beam_indices]
        
        # update input_ids
        token_indices_flat = token_indices.view(-1, 1)
        input_ids = torch.cat([input_ids, token_indices_flat], dim=-1)
        
        # updaate mask
        new_mask = torch.ones((input_ids.shape[0], 1), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
        
        # update scores
        beam_scores = next_scores.view(-1)
        
        cur_len += 1
        
        # termination
        if eos_token_id is not None:
            top_tokens = token_indices[:, 0]
            if (top_tokens == eos_token_id).all():
                break

    # 根据 length_penalty 调整分数
    # Score = LogProb / (Length ** Penalty)
    # 注意：LogProb 是负数。如果 penalty > 0，Length 越大分母越大，数值越小（越接近0），即奖励长句？
    # 通常定义为: score = log_prob / (len^alpha)。因为 log_prob 是负数，除以正数后还是负数。
    # 绝对值变小了（数值变大了）。
    
    final_scores = beam_scores.view(batch_size, num_beams)
    final_scores = final_scores / (cur_len ** length_penalty)
    
    best_beam_idx = torch.argmax(final_scores, dim=-1)
    best_global_idx = best_beam_idx + torch.arange(batch_size, device=device) * num_beams
    return input_ids[best_global_idx]


# --- Test Generation ---

print("#1 贪心解码")
query1 = ["Give me a brief introduction to Shanghai Jiao Tong University.", "介绍一下上海交通大学。", "What is the capital of China?"]
# 如果没有实现附加2，请用循环的方式依次解码query1里的每个字符串并打印出来
for i, response in enumerate(generate(model, query1, max_new_tokens=256, do_sample=False)):
    print(f"[{i}] 问：{query1[i]}\n答：{response}")

print("\n#2 采样解码")
query2 = "Tell me a joke about computers."
for i in range(5):
    response = generate(model, query2, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)    # 可以试试调整这些采样超参数
    print(f"[{i}] 问：{query2}\n答：{response}")

print("\n#3 【附加3】束搜索解码")
query3 = "What is the sum of the first 100 natural numbers? Please think step by step."
response = generate(model, query3, num_beams=4, length_penalty=1.0)
print(f"问：{query3}\n答：{response}")