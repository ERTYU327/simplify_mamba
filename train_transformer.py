import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import TransformerEncoderLayer  # HuggingFace 实现
import time
import os
import math
from data_loader import create_data_loader  # 使用你原有的数据加载函数
import matplotlib.pyplot as plt

class SimpleTransformer(nn.Module):
    """单层 Transformer 语言模型（因果自回归）"""
    def __init__(self, vocab_size, d_model=128, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 可学习位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model) * 0.1)  # 最大长度5000
        # 单层 Transformer Encoder（需手动应用因果掩码）
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 输入形状 (batch, seq, d_model)
        )
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: (batch, seq_len)  token ids
        返回: (batch, seq_len, vocab_size) logits
        """
        seq_len = x.size(1)
        # 嵌入 + 位置编码
        embed = self.embedding(x) * math.sqrt(self.d_model)  # 缩放
        embed = embed + self.pos_encoder[:, :seq_len, :]

        # 生成因果掩码（上三角矩阵，避免看到未来）
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        # TransformerEncoderLayer 接收 mask 参数 (seq_len, seq_len)
        output = self.encoder_layer(embed, src_mask=mask)  # (batch, seq, d_model)
        logits = self.output_layer(output)
        return logits


class TransformerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 创建数据加载器（与原代码一致）
        self.train_loader, self.vocab, self.inv_vocab = create_data_loader(
            config['data_path'],
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            num_workers=config.get('num_workers', 0)
        )
        self.vocab_size = len(self.vocab)
        print(f"词汇表大小: {self.vocab_size}")

        # 创建模型
        self.model = SimpleTransformer(
            vocab_size=self.vocab_size,
            d_model=config['d_model'],
            nhead=config.get('nhead', 8),
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['num_epochs'])

        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.loss_history = []
        self.step_count = []
        # 打印参数量
        print(f"模型参数数量: {self.count_parameters():,}")

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)      # (B, L)
            targets = targets.to(self.device)

            # 前向传播
            logits = self.model(inputs)           # (B, L, V)
            loss = self.criterion(logits.view(-1, self.vocab_size), targets.view(-1))

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['grad_clip'])
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * inputs.numel()
            total_tokens += inputs.numel()

            if batch_idx % self.config['log_interval'] == 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
                print(f'Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {avg_loss:.4f} | PPL: {ppl:.2f}')
                self.loss_history.append(avg_loss)
                self.step_count.append(len(self.loss_history))
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        self.train_losses.append(avg_loss)
        return avg_loss, ppl

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'vocab': self.vocab,
            'config': self.config,
            'train_losses': self.train_losses,
            'loss_history': self.loss_history,
            'step_count': self.step_count
        }
        os.makedirs(self.config['save_dir'], exist_ok=True)
        checkpoint_path = os.path.join(self.config['save_dir'], f'transformer_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'transformer_best.pth')
            torch.save(checkpoint, best_path)
            print(f"新的最佳模型已保存: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.train_losses = checkpoint.get('train_losses', [])  # 兼容旧检查点
        print(f"加载检查点: epoch {self.current_epoch}, loss {self.best_loss:.4f}")

    def train(self):
        print("开始训练 Transformer...")
        try:
         for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            start_time = time.time()
            avg_loss, ppl = self.train_epoch()
            self.scheduler.step()
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch} 完成 | 时间: {epoch_time:.2f}s | 平均损失: {avg_loss:.4f} | 困惑度: {ppl:.2f}')

            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss

            if epoch % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, avg_loss, is_best)
        except KeyboardInterrupt:
            print("\n训练被中断，保存当前状态...")
            self.save_checkpoint(self.current_epoch, self.best_loss)
            if self.train_losses:
                    self.plot_losses()
            else:
                    print("训练过早中断，尚无完整的 epoch 损失数据，跳过损失曲线绘制。")
        finally:
            # 无论正常结束还是中断，都绘制损失曲线
            self.plot_losses()

    def generate_text(self, prompt, max_length=100, temperature=0.8):
        """简易生成函数（每步重新计算整个序列）"""
        self.model.eval()
        prompt = self.preprocess_text(prompt)
        tokens = [self.vocab.get(char, 0) for char in prompt]
        generated = tokens.copy()

        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([generated[-self.config['seq_len']:]], dtype=torch.long).to(self.device)
                logits = self.model(input_tensor)  # (1, seq_len, V)
                next_token_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                if next_token == 0:
                    break

        generated_text = ''.join([self.inv_vocab.get(token, '') for token in generated])
        return generated_text

    def plot_losses(self):
        """绘制训练损失曲线（包括batch级和epoch级）"""
        if not self.loss_history and not self.train_losses:
            print("没有损失数据可绘制")
            return

        plt.figure(figsize=(12, 5))

        if self.loss_history:
            plt.subplot(1, 2, 1)
            plt.plot(self.step_count, self.loss_history, label='Batch Loss', alpha=0.7)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Loss per Step')
            plt.legend()
            plt.grid(True)

        if self.train_losses:
            plt.subplot(1, 2, 2)
            plt.plot(range(len(self.train_losses)), self.train_losses, marker='o', label='Epoch Avg Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss per Epoch')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.config['save_dir'], 'transformer_loss_curve.png')
        plt.savefig(save_path, dpi=150)
        print(f"损失曲线已保存至: {save_path}")
        plt.show()
    def preprocess_text(self, text):
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s，。！？；：、（）【】《》]', '', text)
        return text.strip()


def main():
    # 训练配置（与原代码完全一致，仅增加 nhead 和 dim_feedforward）
    config = {
        'data_path': '都市极品医神.txt',   # 请修改为你的文件路径
        'batch_size': 16,
        'seq_len': 256,
        'd_model': 256,
        'nhead': 8,                       # 注意力头数
        'dim_feedforward': 512,            # 前馈网络维度
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'num_epochs': 100,
        'log_interval': 1,
        'save_interval': 5,
        'save_dir': 'transformer_checkpoints',
        'num_workers': 0
    }

    trainer = TransformerTrainer(config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被中断，保存检查点...")
        trainer.save_checkpoint(trainer.current_epoch, trainer.best_loss)

    print("\n测试文本生成...")
    prompt = "江城高铁站"
    generated = trainer.generate_text(prompt, max_length=50)
    print(f"提示: {prompt}")
    print(f"生成: {generated}")


if __name__ == "__main__":
    main()