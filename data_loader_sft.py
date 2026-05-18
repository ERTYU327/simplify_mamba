import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List
from modelscope.msdatasets import MsDataset

class SFTDataset(Dataset):
    def __init__(self, dataset_name: str, tokenizer: AutoTokenizer, max_length: int = 866):
        """
        Args:
            dataset_name: Hugging Face 数据集名称或本地路径
            tokenizer: 预训练 tokenizer（如 'bert-base-chinese' 或 'Qwen/Qwen-7B'）
            max_length: 最大序列长度（超过则截断）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(dataset_name)

    def _load_data(self, dataset_name):
        print(f"正在加载 SFT 数据集: {dataset_name}")
        #ds = load_dataset("train_data", split="train[:3000000]")
        ds = load_dataset("train_data",  split='train')
        print(f'下载完成，共 {len(ds)} 条原始数据')

        samples = []
        for idx, item in enumerate(ds):
            conversations = item.get('conversations', [])
            if not conversations:
                continue
            instruction = None
            output = None
            for msg in conversations:
                if msg.get('from') == 'human':
                    instruction = msg.get('value', '')
                elif msg.get('from') == 'assistant':
                    output = msg.get('value', '')
                    if instruction is not None and output is not None:
                        break

            if instruction is None or output is None:
                continue

            samples.append({
                'instruction': instruction,
                'input': '',  # 没有单独的 input 字段
                'output': output
            })

        print(f"加载了 {len(samples)} 条有效样本")
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample['instruction']
        if sample['input']:
            prompt += "\n" + sample['input']
        response = sample['output']
        prompt_enc = self.tokenizer(prompt, add_special_tokens=False)
        prompt_ids = prompt_enc['input_ids']
        response_enc = self.tokenizer(response, add_special_tokens=False)
        response_ids = response_enc['input_ids']
        #prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        #response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|sep|>']})
        #prompt_len = len(prompt_ids)
        if self.tokenizer.eos_token_id is not None:
            response_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + response_ids

        response = prompt_ids + response_ids
        #response = [-100] * len(prompt_ids) + response_ids
        return {
            'input_ids': input_ids,
            'response': response,
            'attention_mask': [1] * len(input_ids) ,
        }
    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[。!？!?\n])', text)
        return [s.strip() for s in sentences if s.strip()]
def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer, max_length: int):
    """
    动态 padding 和截断，生成 attention_mask，并返回张量。
    """
    input_ids_list = []
    labels_list = []
    for item in batch:
        #input_ids = item['input_ids'][:max_length]  # 截断
        input_ids = item["input_ids"]
        #response = item['response'][:max_length]
        response = item["response"]
        input_ids_list.append(input_ids)
        labels_list.append(response)

    max_len = max(len(ids) for ids in input_ids_list)
    #max_len = min(max_len, max_length)


    padded_input = 0
    padded_response = 0
    padded_input_ids_list = []
    padded_labels_list = []
    #attention_masks = []
    for input_ids, labels in zip(input_ids_list, labels_list):
        pad_len = max_len - len(input_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded_input_ids = torch.tensor(input_ids + [pad_id] * pad_len)
        padded_input_ids_list.append(padded_input_ids)
        padded_labels = torch.tensor(labels + [-100] * pad_len)
        padded_labels_list.append(padded_labels)
        #attention_masks.append([1] * len(input_ids) + [0] * pad_len)
    padded_input = torch.stack(padded_input_ids_list,dim=0)
    padded_response = torch.stack(padded_labels_list,dim=0)
    return {
        'input_ids': padded_input,
        'response': padded_response,
        #'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
    }

def create_sft_dataloader(dataset_name, tokenizer,
                          batch_size, max_length, num_workers):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'
    #tokenizer.padding_side = 'right'

    dataset = SFTDataset(dataset_name, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length),
        pin_memory=True
    )
    return dataloader, tokenizer