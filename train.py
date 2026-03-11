import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import math
from data_loader import create_data_loader
from mamba import MAMBA


class MambaTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建数据加载器
        self.train_loader, self.vocab, self.inv_vocab = create_data_loader(
            config['data_path'],
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            num_workers=config.get('num_workers', 0)
        )
        
        # 创建模型
        self.model = MAMBA(
            d_model=config['d_model'],
            d_state=config['d_state']
        ).to(self.device)
        
        # 创建词嵌入层
        self.vocab_size = len(self.vocab)
        self.embedding = nn.Embedding(
            self.vocab_size, 
            config['d_model']
        ).to(self.device)
        
        # 创建输出层
        self.output_layer = nn.Linear(
            config['d_model'], 
            self.vocab_size
        ).to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略未知字符
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_layer.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config['num_epochs']
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        print(f"模型参数数量: {self.count_parameters():,}")
        print(f"词汇表大小: {self.vocab_size}")

    def count_parameters(self):
        return sum(p.numel() for p in list(self.model.parameters())
                   + list(self.embedding.parameters())
                   + list(self.output_layer.parameters()) if p.requires_grad)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # 移动到设备
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            embeddings = self.embedding(inputs)  # (B, L, D)
            
            # 检查维度
            batch_size, seq_len, d_model = embeddings.shape
            assert d_model == self.config['d_model'], f"嵌入维度不匹配: {d_model} != {self.config['d_model']}"
            
            # 通过Mamba模型
            outputs = self.model(embeddings)  # (B, L, D)
            
            # 检查输出维度
            assert outputs.shape == embeddings.shape, f"输出维度不匹配: {outputs.shape} != {embeddings.shape}"
            
            # 通过输出层
            logits = self.output_layer(outputs)  # (B, L, vocab_size)
            
            # 计算损失
            loss = self.criterion(
                logits.view(-1, self.vocab_size), 
                targets.view(-1)
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_layer.parameters()),
                max_norm=self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * inputs.numel()
            total_tokens += inputs.numel()
            
            if batch_idx % self.config['log_interval'] == 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
                print(f'Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {avg_loss:.4f} | PPL: {ppl:.2f}')
        
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        
        return avg_loss, ppl
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'embedding_state_dict': self.embedding.state_dict(),
            'output_layer_state_dict': self.output_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'vocab': self.vocab,
            'config': self.config
        }
        
        # 保存检查点
        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最好结果，保存为最佳模型
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"新的最佳模型已保存: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        self.output_layer.load_state_dict(checkpoint['output_layer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        print(f"加载检查点: epoch {self.current_epoch}, loss {self.best_loss:.4f}")
    
    def train(self):
        print("开始训练...")
        
        # 创建保存目录
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 训练一个epoch
            avg_loss, ppl = self.train_epoch()
            
            # 更新学习率
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch} 完成 | 时间: {epoch_time:.2f}s | '
                  f'平均损失: {avg_loss:.4f} | 困惑度: {ppl:.2f}')
            
            # 保存检查点
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            if epoch % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, avg_loss, is_best)
    
    def generate_text(self, prompt, max_length=100, temperature=0.8):
        """生成文本"""
        self.model.eval()
        
        # 预处理提示
        prompt = self.preprocess_text(prompt)
        tokens = [self.vocab.get(char, 0) for char in prompt]
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            # 初始化状态
            h = None
            
            for i in range(max_length):
                # 准备输入
                input_tensor = torch.tensor([generated_tokens[-self.config['seq_len']:]], 
                                          dtype=torch.long).to(self.device)
                
                # 嵌入
                embeddings = self.embedding(input_tensor)
                
                # 单步推理
                if h is None:
                    output, h = self.model.step(embeddings[:, -1, :])
                else:
                    output, h = self.model.step(embeddings[:, -1, :], h)
                
                # 获取下一个token的概率
                logits = self.output_layer(output)
                probs = torch.softmax(logits / temperature, dim=-1)
                
                # 采样
                next_token = torch.multinomial(probs[0], 1).item()
                generated_tokens.append(next_token)
                
                # 如果遇到结束标记，停止生成
                if next_token == 0:  # 未知字符作为结束标记
                    break
        
        # 转换为文本
        generated_text = ''.join([self.inv_vocab.get(token, '') for token in generated_tokens])
        return generated_text
    
    def preprocess_text(self, text):
        """预处理文本"""
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s，。！？；：、（）【】《》]', '', text)
        return text.strip()


def main():
    # 训练配置
    config = {
        'data_path': '都市极品医神.txt',
        'batch_size': 16,
        'seq_len': 256,
        'd_model': 384,
        'd_state': 384,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1,
        'num_epochs': 100,
        'log_interval': 1,
        'save_interval': 5,
        'save_dir': 'checkpoints',
        'num_workers': 0
    }
    
    # 创建训练器
    trainer = MambaTrainer(config)
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被中断，保存检查点...")
        trainer.save_checkpoint(trainer.current_epoch, trainer.best_loss)
    
    # 测试文本生成
    print("\n测试文本生成...")
    prompt = "江城高铁站"
    generated = trainer.generate_text(prompt, max_length=50)
    print(f"提示: {prompt}")
    print(f"生成: {generated}")


if __name__ == "__main__":
    main()