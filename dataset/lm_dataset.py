from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json',data_files = data_path,split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        input_ids = self.tokenizer(
            str(sample['text']),
            add_special_tokens = False,
            max_length = self.max_length - 2, # BOS 与 EOS 占据两个位置
            truncation = True,
        ).input_ids

        tokens = [self.tokenizer.bos_token] + input_ids + [self.tokenizer.eos_token]

        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        input_ids = torch.tensor(input_ids,dtype=torch.long)
        labels = input_ids.clone() # 这里不用考虑移位是因为我们在模型内部计算 loss 的时候做了这个操作

        # 条件生成布尔掩码
        labels[input_ids == self.tokenizer.pad_token_id] = -100
       
        return input_ids,labels

class SFTDataset(Dataset):
    def __init__(self,jsonl_path,tokenizer,max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        features = Features({'conversations':[{'role':Value('string'),'content':Value('string'),'reasoning_content:':Value('string'),'tools':Value('string'),'tool_calls':Value('string')}],})

        self.samples = load_dataset('json',data_files = jsonl_path,split='train',features=features)

        # 提取关键的 token ids
        self.bos_id = self.tokenizer(f'{tokenizer.bos_token}assistant\n',add_special_tokens = False).input_ids
        self.eos_id = self.tokenizer(f'{tokenizer.eos_token}\n',add_special_tokens = False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self,conversations):
        messages = []
        tools = []

        for message in conversations:
            message = dict(message)

            if message.get('role') == 'system' and message.get('tools'):
                tools = json.loads(message['tools']) if isinstance(message['tools'],str) else message['tools']

            if message.get('tool_calls') and isinstance(message['tool_calls'],str):
                message['tool_calls'] = json.loads(message['tool_calls'])

            messages.append(message)

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=False,
            tools = tools
        )

    def generate_labels(self,input_ids):

        labels = [-100] * len(input_ids)
        i = 0

        while i < len(input_ids):
            # 寻找 LLM 回复的起点
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                # 寻找终点
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                for j in range(start,min(end+len(self.eos_id),self.max_length)):
                    labels[j] = input_ids[j]
                # 更新指针，跳过已经处理完的区间 - 可能存在多轮对话
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i = i + 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # 1. 预处理，拼模版，后处理
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)

        # 2. tokenize并且截断
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # 3. Padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 4. 生成 lables
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids,dtype=torch.long), torch.tensor(labels,dtype=torch.long)