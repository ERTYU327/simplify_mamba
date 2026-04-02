import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from datasets import load_dataset

class TextDataset(Dataset):
    def __init__(self, file_path, seq_len=256, vocab_size=81920):
        self.seq_len = seq_len
        self.vocab_size = vocab_size


        dataset_name = file_path
        self._load_from_hf(dataset_name)

    def _load_from_hf(self, dataset_name):
        print(f"正在加载 Hugging Face 数据集: {dataset_name}")
        ds = load_dataset(
            "Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
            cache_dir="/Users/a123/mamba/train_data",
        )
        print('已缓存')
        if 'train' in ds:
            data = ds['train']
        else:
            data = ds[list(ds.keys())[0]]  # 取第一个 split

        texts = []
        for item in data:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')

            combined = f"{instruction}\n{input_text}\n{output}\n"
            texts.append(combined)

        full_text = ''.join(texts)
        full_text = self.preprocess_text(full_text)

        # 分词
        self.tokens = self.tokenize(full_text)
        self.vocab, self.inv_vocab = self.build_vocab(self.tokens)
        self.token_ids = self.tokens_to_ids(self.tokens)
        print(f"唯一词数: {len(self.tokens)}")
        print(f"数据集总词数: {len(self.token_ids)}")
        print(f"词汇表大小: {len(self.vocab)}")

    def preprocess_text(self, text):
        """基础清洗，可保留标点符号"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s，。！？；：、（）【】《》]', '', text)
        return text.strip()

    def tokenize(self, text):
        #from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        return list(jieba.cut(text, cut_all=False))
        #return tokenizer.tokenize(text)

    def build_vocab(self, tokens):
        """基于词列表构建词汇表，取频率最高的 vocab_size-1 个词"""
        counter = Counter(tokens)
        total_tokens = len(self.tokens)
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 测试不同词汇表大小的覆盖率
        for vocab_size in [20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000]:
            covered = sum(cnt for _, cnt in sorted_words[:vocab_size - 1])  # 留一个给 <unk>
            coverage = covered / total_tokens
            print(f"vocab_size={vocab_size}: coverage={coverage:.2%}")
        #most_common = counter.most_common(len(tokens) - 1)
        most_common = counter.most_common(self.vocab_size - 1)
        vocab = {word: idx+1 for idx, (word, _) in enumerate(most_common)}
        vocab['<unk>'] = 0
        inv_vocab = {idx: word for word, idx in vocab.items()}
        return vocab, inv_vocab

    def tokens_to_ids(self, tokens):
        """将词列表转换为 ID 序列，未知词用 0 填充"""
        return [self.vocab.get(token, 0) for token in tokens]

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.token_ids[idx:idx+self.seq_len]
        target_seq = self.token_ids[idx+1:idx+self.seq_len+1]
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        return input_tensor, target_tensor


def create_data_loader(file_path, batch_size=32, seq_len=256, num_workers=0):
    dataset = TextDataset(file_path, seq_len=seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, dataset.vocab, dataset.inv_vocab