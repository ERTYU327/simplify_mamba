import torch
from torch.utils.data import Dataset, DataLoader
import re

class TextDataset(Dataset):
    def __init__(self, file_path, seq_len=256, vocab_size=5000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 简单的文本预处理
        text = self.preprocess_text(text)
        
        # 构建词汇表
        self.vocab, self.inv_vocab = self.build_vocab(text)
        
        # 将文本转换为token序列
        self.tokens = self.text_to_tokens(text)
        
        print(f"数据集大小: {len(self.tokens)} tokens")
        print(f"词汇表大小: {len(self.vocab)}")
    
    def preprocess_text(self, text):
        # 移除多余的空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符，保留中文和基本标点
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s，。！？；：、（）【】《》]', '', text)
        return text.strip()
    
    def build_vocab(self, text):
        # 统计字符频率
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # 选择频率最高的字符作为词汇表
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        vocab = {char: idx + 1 for idx, (char, _) in enumerate(sorted_chars[:self.vocab_size-1])}
        vocab['<unk>'] = 0  # 未知字符
        
        inv_vocab = {idx: char for char, idx in vocab.items()}
        
        return vocab, inv_vocab
    
    def text_to_tokens(self, text):
        tokens = []
        for char in text:
            tokens.append(self.vocab.get(char, 0))  # 0表示未知字符
        return tokens
    
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        # 获取输入序列和目标序列
        input_seq = self.tokens[idx:idx + self.seq_len]
        target_seq = self.tokens[idx + 1:idx + self.seq_len + 1]
        
        # 转换为tensor
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


if __name__ == "__main__":
    # 测试数据加载器
    dataloader, vocab, inv_vocab = create_data_loader(
        "都市极品医神.txt", 
        batch_size=2, 
        seq_len=64
    )
    
    print("测试数据加载器...")
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        
        # 显示前几个字符
        sample_text = ''.join([inv_vocab.get(token.item(), '<unk>') for token in inputs[0][:10]])
        print(f"  Sample text: {sample_text}")
        
        if i >= 2:  # 只显示前3个batch
            break