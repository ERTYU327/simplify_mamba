import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
from data_loader_conversation import create_data_loader
from LM import LM
import matplotlib.pyplot as plt
import transformers
class BaijuTrainer:
    def __init__(self, config):
        self.config = config
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.train_loader, self.vocab, self.inv_vocab = create_data_loader(
            config['data_path'],
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            num_workers=config.get('num_workers', 0)
        )
        val_ratio = config.get('val_ratio', 5e-4)
        self.train_loader, self.val_loader = self.split_train_val(self.train_loader, val_ratio)

        self.model = LM(
            d_model=config['d_model'],
            d_state=config['d_state'],
            first_block=config['first_block'],
            other_block=config['other_block'],
        ).to(self.device)
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

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略未知字符
        #self.optimizer = optim.Adafactor(
            #list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_layer.parameters()),
            #lr=config['learning_rate'],
            #weight_decay=config['weight_decay']
        #)
        self.optimizer = transformers.Adafactor(
            list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_layer.parameters()),
            lr=None,  # 让 Adafactor 自动管理学习率
            eps=(1e-30, 1e-3),  # 默认的 eps 参数
            clip_threshold=1.0,  # 梯度裁剪阈值
            decay_rate=-0.8,  # 相对学习率的衰减率（负值表示使用固定衰减）
            scale_parameter=True,  # 启用参数缩放
            relative_step=True,  # 使用相对步长（自动调整学习率）
            warmup_init=False,  # 是否预热初始化
        )
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #self.optimizer,
            #T_max=config['num_epochs']
            #5000
        #)
        self.auto_reg_ratio = 0.0  # 初始全 teacher forcing
        self.auto_reg_increment = 0.05  # 每个 epoch 增加比例
        self.total_epochs = config['num_epochs']  # 用于计算最终比例
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []

        self.loss_history = []
        self.step_count = []
        print(f"模型参数数量: {self.count_parameters():,}")
        print(f"词汇表大小: {self.vocab_size}")

    def count_parameters(self):
        return sum(p.numel() for p in list(self.model.parameters()))

    def split_train_val(self, train_loader, val_ratio):
        """从原始 train_loader 中拆分出验证集"""
        # 注意：train_loader 是 DataLoader，需要获取其数据集并重新分割
        dataset = train_loader.dataset
        total_len = len(dataset)
        val_len = int(total_len * val_ratio)
        train_len = total_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True,
            num_workers=self.config['num_workers'], pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True
        )
        return train_loader, val_loader

    def validate_teacher_forcing(self):
        """使用教师强制模式计算验证集损失"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                embeddings = self.embedding(inputs)
                outputs = self.model(embeddings)
                logits = self.output_layer(outputs)
                loss = self.criterion(logits.view(-1, self.vocab_size), targets.view(-1))
                total_loss += loss.item() * inputs.numel()
                total_tokens += inputs.numel()
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        self.model.train()
        return avg_loss, ppl
    def train_epoch_autoregressive(self):
        """使用自回归方式训练一个 epoch（逐个 token 预测）"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)  # (B, L)
            targets = targets.to(self.device)  # (B, L)  shift 后的目标
            B, L = inputs.shape

            # 初始化状态为 None
            h = None
            loss = 0.0

            # 逐个时间步
            for t in range(L):
                # 当前 token
                x_t = inputs[:, t]  # (B,)
                x_emb = self.embedding(x_t)  # (B, 1, D)

                if h is None:
                    y, h = self.model.step(x_emb)  # y: (B, 1, D)
                else:
                    y, h = self.model.step(x_emb, h)

                logits = self.output_layer(y)  # (B, vocab_size)
                target_t = targets[:, t]  # (B,)
                step_loss = self.criterion(logits, target_t)
                loss += step_loss

            # 平均每个位置的损失
            loss = loss / L

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.embedding.parameters()) +
                list(self.output_layer.parameters()),
                max_norm=self.config['grad_clip']
            )
            self.optimizer.step()
            #self.scheduler.step(loss)

            total_loss += loss.item() * B * L
            total_tokens += B * L

            if batch_idx % self.config['log_interval'] == 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
                print(f'[AR] Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {avg_loss:.4f} | 'f'PPL: {ppl:.2f}')

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return avg_loss, ppl

    def train_epoch(self, epoch):
        """一个 epoch 的训练，根据 auto_reg_ratio 决定使用 teacher forcing 还是自回归"""
        # 随机决定这个 epoch 的训练模式
        #use_ar = torch.rand(1).item() < self.auto_reg_ratio

        #if use_ar:
            #print(f"Epoch {self.current_epoch}: 使用自回归训练")
            #return self.train_epoch_autoregressive()
        #else:
            #print(f"Epoch {self.current_epoch}: 使用 teacher forcing 训练")
            # 调用原有的 teacher forcing 训练逻辑（你之前写好的）
            #return self.train_epoch_teacher_forcing()
        if epoch == 0 :
            return self.train_epoch_teacher_forcing()
        else:
            return self.train_epoch_teacher_forcing()

    def train_epoch_teacher_forcing(self):
        """传统的 teacher forcing 训练"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            embeddings = self.embedding(inputs)
            outputs = self.model(embeddings)  # (B, L, D)
            logits = self.output_layer(outputs)  # (B, L, vocab_size)

            loss = self.criterion(logits.view(-1, self.vocab_size), targets.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.embedding.parameters()) + list(
                    self.output_layer.parameters()),
                max_norm=self.config['grad_clip']
            )
            self.optimizer.step()
            #self.scheduler.step(loss)

            total_loss += loss.item() * inputs.numel()
            total_tokens += inputs.numel()

            if batch_idx % self.config['log_interval'] == 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
                print(f'[TF] Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {avg_loss:.4f} | PPL: {ppl:.2f}')
            if batch_idx % (self.config['log_interval'] * 1000) == 0 and batch_idx > 0:
                val_loss, val_ppl = self.validate_teacher_forcing()
                print(f'val_Loss: {val_loss:.2f} | val_PPL: {val_ppl:.2f}')

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
            #'scheduler_state_dict': self.scheduler.state_dict(),
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
        #self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        print(f"加载检查点: epoch {self.current_epoch}, loss {self.best_loss:.4f}")
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

    def train(self):
        print("开始训练...")
        os.makedirs(self.config['save_dir'], exist_ok=True)

        # 设置初始比例
        self.auto_reg_ratio = 0.4
        increment = self.auto_reg_increment

        try:
            for epoch in range(self.current_epoch, self.config['num_epochs']):
                self.current_epoch = epoch
                start_time = time.time()

                # 训练一个 epoch
                avg_loss, ppl = self.train_epoch(epoch)

                epoch_time = time.time() - start_time
                print(f'Epoch {epoch} 完成 | 时间: {epoch_time:.2f}s | '
                      f'平均损失: {avg_loss:.4f} | 困惑度: {ppl:.2f} | '
                      f'自回归比例: {self.auto_reg_ratio:.2f}')

                # 更新自回归比例（线性增加，最大1.0）
                self.auto_reg_ratio = min(1.0, self.auto_reg_ratio + increment)

                # 保存检查点等...
                is_best = avg_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_loss
                if epoch % self.config['save_interval'] == 0 or is_best:
                    self.save_checkpoint(epoch, avg_loss, is_best)
        except KeyboardInterrupt:
            print("\n训练被中断，保存当前状态...")
            self.save_checkpoint(self.current_epoch, self.best_loss)
        finally:
            self.plot_losses()

    def generate_text(self, prompt, max_length=100, temperature=1.0):
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
                    y, h = self.model.step(embeddings[:,-1,:])
                else:
                    y, h = self.model.step(embeddings[:,-1,:], h)
                # 获取下一个token的概率
                logits = self.output_layer(y)
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

    def generate_text1(self, prompt, max_length=100, temperature=1.0):
        """自回归生成文本"""
        self.model.eval()
        prompt = self.preprocess_text(prompt)
        tokens = [self.vocab.get(char, 0) for char in prompt]
        generated = tokens.copy()
        seq_len = self.config['seq_len']

        with torch.no_grad():
            for _ in range(max_length):
                # 取最后 seq_len 个 token 作为输入
                input_tokens = generated[-seq_len:]
                input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)
                embeddings = self.embedding(input_tensor)  # (1, L, d_model)
                outputs = self.model(embeddings)  # (1, L, d_model)
                logits = self.output_layer(outputs)  # (1, L, vocab_size)
                last_logits = logits[0, -1, :]  # (vocab_size,)
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                if next_token == 0:  # 假设 0 是结束标记（未知字符）
                    break

        generated_text = ''.join([self.inv_vocab.get(t, '') for t in generated])
        return generated_text
    def generate_text2(self, prompt, max_length=50, temperature=1.0):
        """生成文本（单步预测，仅用于测试）"""
        self.model.eval()

        # 预处理提示
        prompt = self.preprocess_text(prompt)
        tokens = [self.vocab.get(char, 0) for char in prompt]

        with torch.no_grad():
            # 准备输入（取最后 seq_len 个 token）
            input_tensor = torch.tensor([tokens[-self.config['seq_len']:]], dtype=torch.long).to(self.device)
            embeddings = self.embedding(input_tensor)  # (1, seq_len, d_model)
            output = self.model(embeddings)  # (1, seq_len, d_model)
            logits = self.output_layer(output)  # (1, seq_len, vocab_size)

            # 取最后一个时间步的 logits
            last_logits = logits[0, -1, :]  # (vocab_size,)
            probs = torch.softmax(last_logits / temperature, dim=-1)

            # 采样下一个 token
            next_token = torch.multinomial(probs, 1).item()

            # 生成文本（原提示 + 新 token）
            generated_tokens = tokens + [next_token]
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
        'data_path': 'Congliu/Chinese-DeepSeek-R1-Distill-data-110k',
        'batch_size': 2,
        'seq_len': 256,
        'd_model': 256,
        'd_state': 256,
        'learning_rate': 1e-2,
        'weight_decay': 1e-4,
        'grad_clip': 1,
        'num_epochs': 1,
        'log_interval': 1,
        'save_interval': 5,
        'save_dir': 'checkpoints',
        'num_workers': 0,
        'first_block':4,
        'other_block':8,
    }

    # 创建训练器
    trainer = BaijuTrainer(config)

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被中断，保存检查点...")
        trainer.save_checkpoint(trainer.current_epoch, trainer.best_loss)

    # 测试文本生成
    print("\n测试文本生成...")
    prompt = "为什么白天看不到星星。"
    generated = trainer.generate_text(prompt, max_length=100,temperature=1.0)
    print(f"提示: {prompt}")
    print(f"生成: {generated}")

def eval(prompt, max_length=100, temperature=1.0):
    config = {
        'data_path': 'Congliu/Chinese-DeepSeek-R1-Distill-data-110k',
        'batch_size': 2,
        'seq_len': 256,
        'd_model': 256,
        'd_state': 256,
        'learning_rate': 1e-2,
        'weight_decay': 1e-4,
        'grad_clip': 1,
        'num_epochs': 1,
        'log_interval': 1,
        'save_interval': 5,
        'save_dir': 'checkpoints',
        'num_workers': 0,
        'first_block':4,
        'other_block':8,
    }

    # 创建训练器
    trainer = BaijuTrainer(config)
    trainer.load_checkpoint('checkpoints/best_model.pth')
    trainer.model.eval()
    prompt = trainer.preprocess_text(prompt)
    tokens = [trainer.vocab.get(char, 0) for char in prompt]

    generated_tokens = tokens.copy()

    with torch.no_grad():
        # 初始化状态
        h = None

        for i in range(max_length):
            # 准备输入
            input_tensor = torch.tensor([generated_tokens[-trainer.config['seq_len']:]],
                                        dtype=torch.long).to(trainer.device)

            # 嵌入
            embeddings = trainer.embedding(input_tensor)

            # 单步推理
            if h is None:
                output, h = trainer.model.step_generate(embeddings[:,-1,:])
            else:
                output, h = trainer.model.step_generate(embeddings[:,-1,:], h)

            # 获取下一个token的概率
            logits = trainer.output_layer(output)
            probs = torch.softmax(logits / temperature, dim=-1)

            # 采样
            next_token = torch.multinomial(probs[0], 1).item()
            generated_tokens.append(next_token)

            # 如果遇到结束标记，停止生成
            if next_token == 0:  # 未知字符作为结束标记
                break

    # 转换为文本
    generated_text = ''.join([trainer.inv_vocab.get(token, '') for token in generated_tokens])
    return generated_text

if __name__ == "__main__":
    main()
    prompt = "为什么白天看不到星星。"
    evaluate = eval(prompt, max_length=100, temperature=1.0)
    print(f"提示: {prompt}")
    print(f"生成: {evaluate}")
