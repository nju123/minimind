from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value


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
