import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(f'开始加载')
from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
#tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
print('加载完成')
TOKENIZER = tokenizer
import math
import time
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import transformers
from torch.nn import functional as F
from BaijuFlex import BaijuFlex
from LM import LM
from data_loader_sft import create_sft_dataloader
from torch.utils.data import DataLoader
class BaijuTrainer:
    def __init__(self, config):
        self.config = config
        #if torch.backends.mps.is_available():
            #self.device = torch.device("mps")
        #else:
        self.device = torch.device("cpu")
        self.train_loader, tokenizer = create_sft_dataloader(
            dataset_name=config['data_path'],
            tokenizer=TOKENIZER,
            batch_size=config['batch_size'],
            #max_length=config['seq_len'],
            max_length=866,
            num_workers=config.get('num_workers', 0)
        )
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        val_ratio = config.get('val_ratio', 5e-4)
        self.train_loader, self.val_loader = self.split_train_val(self.train_loader, val_ratio)
        self.model = BaijuFlex(
            d_model=config['d_model'],
            d_state=config['d_state'],
            num_blocks=config['num_blocks'],
        ).to(self.device)
        self.embedding = nn.Embedding(self.vocab_size, config['d_model']).to(self.device)
        self.output_layer = nn.Linear(config['d_model'], self.vocab_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1).to(self.device)
        #self.optimizer = transformers.Adafactor(
            #list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_layer.parameters()),
            #lr=None,  # 让 Adafactor 自动管理学习率
            #eps=(1e-30, 1e-3),  # 默认的 eps 参数
            #clip_threshold=1.0,  # 梯度裁剪阈值
            #decay_rate=-0.8,  # 相对学习率的衰减率（负值表示使用固定衰减）
            #scale_parameter=True,  # 启用参数缩放
            #relative_step=True,  # 使用相对步长（自动调整学习率）
            #warmup_init=True,  # 是否预热初始化
        #)
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.embedding.parameters()) + list(self.output_layer.parameters())
                                           , lr=config['learning_rate'],
                                           weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=200, T_mult=2, eta_min=1e-6)
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

    def warmup_cosine(self, step, warmup_steps, total_steps, peak_lr=1e-4, min_lr=1e-6):
        if step < warmup_steps:
            return step / warmup_steps  # 线性增加
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

    def count_parameters(self):
        return sum(p.numel() for p in list(self.model.parameters())
         + list(self.embedding.parameters())
         + list(self.output_layer.parameters()) if p.requires_grad)

    def split_train_val(self, train_loader, val_ratio):
        dataset = train_loader.dataset
        total_len = len(dataset)
        val_len = int(total_len * val_ratio)
        train_len = total_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

        from data_loader_conversation2 import collate_fn
        train_loader = DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True,
            num_workers=self.config['num_workers'], pin_memory=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, self.config['seq_len'])
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, self.config['seq_len'])
        )
        return train_loader, val_loader

    def generate_text_old(self, prompt, max_length=100, temperature=1.0,top_k=20,top_p=None,):
        """使用 tokenizer 生成文本"""
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = input_ids.copy()
        with torch.no_grad():
            h = None
            for _ in range(max_length):
                ctx = generated_ids[-self.config['seq_len']:]
                ctx_tensor = torch.tensor([ctx], dtype=torch.long).to(self.device)
                ctx_emb = self.embedding(ctx_tensor)
                if h is None:
                    output, h = self.model.step(ctx_emb[:, -1:, :], max_length)
                else:
                    output, h = self.model.step(ctx_emb[:, -1:, :], h)
                logits = self.output_layer(output)[0] / temperature
                probs = torch.softmax(logits, dim=-1)
                if top_p is not None and top_p < 1.0:
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)  # (1, vocab_size)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = -float('Inf')
                    logits = logits.squeeze(0)  # 恢复一维
                    probs = torch.softmax(logits, dim=-1)
                if top_k:
                    top_probs, top_indices = torch.topk(probs, top_k)
                    top_probs = top_probs / top_probs.sum()
                    next_id = top_indices[torch.multinomial(top_probs, 1)].item()
                else:
                   next_id = torch.multinomial(probs, 1).item()
                generated_ids.append(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    break
        self.model.train()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    def generate_text_new(self, prompt, max_length, temperature=1.0,top_k=20,top_p=None,):
        """使用 tokenizer 生成文本"""
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = input_ids.copy()
        with torch.no_grad():

            ctx = generated_ids[-self.config['seq_len']:]
            ctx_tensor = torch.tensor([ctx], dtype=torch.long).to(self.device)
            ctx_emb = self.embedding(ctx_tensor)
            output = self.model.step(ctx_emb,max_length)
            logits = self.output_layer(output)[0] / temperature
            probs = torch.softmax(logits, dim=-1)
            for i in range(max_length):
                next_id = torch.multinomial(probs[i], 1).item()
                generated_ids.append(next_id)
        self.model.train()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    def generate_text(self, prompt, max_length=100, temperature=1.0, top_p=None):
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = input_ids.copy()
        with torch.no_grad():
            #h = None
            #for _ in range(max_length):
            ctx = generated_ids[-self.config['seq_len']:]
            ctx_tensor = torch.tensor([ctx], dtype=torch.long).to(self.device)
            ctx_emb = self.embedding(ctx_tensor)
            #if h is None:
                #output, h = self.model.step(ctx_emb[:, -1:, :])
            #else:
                #output, h = self.model.step(ctx_emb[:, -1:, :], h)
            output = self.model.step(ctx_emb,max_length)
            logits = self.output_layer(output)[0] / temperature  # (vocab_size,)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # 至少保留一个 token
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            #next_id = torch.multinomial(probs, 1).item()
            generated_ids = torch.multinomial(probs, 1).item()
            #generated_ids.append(next_id)
            #if next_id == self.tokenizer.eos_token_id:
                    #break
        self.model.train()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    def validate_teacher_forcing(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                embeddings = self.embedding(input_ids)
                outputs = self.model(embeddings)
                logits = self.output_layer(outputs)
                loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
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

            h = None
            loss = 0.0

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

    def validate_epoch_autoregressive(self, max_gen_len=16):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)  # (B, L)
                B, L = input_ids.shape
                gen_steps = min(max_gen_len, L - 1)
                if gen_steps <= 0:
                    continue

                h = None
                ctx = input_ids[:, :1]  # (B, 1)
                loss = 0.0
                for step in range(gen_steps):
                    x_t = ctx[:, -1:]  # (B, 1)
                    x_emb = self.embedding(x_t)
                    if h is None:
                        y, h = self.model.step(x_emb)
                    else:
                        y, h = self.model.step(x_emb, h)
                    logits = self.output_layer(y)  # (B, vocab_size)
                    target = input_ids[:, step + 1]  # (B,)
                    step_loss = self.criterion(logits, target)
                    loss += step_loss
                    # 自回归：使用预测的 token（argmax）作为下一步输入
                    pred_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
                    ctx = torch.cat([ctx, pred_token], dim=1)
                loss = loss / gen_steps
                total_loss += loss.item() * B
                total_tokens += B

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        self.model.train()
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
            # 调用原有的 teacher forcing 训练逻辑
            #return self.train_epoch_teacher_forcing()
        if epoch == 0 :
            return self.train_epoch_teacher_forcing()
        else:
            return self.train_epoch_teacher_forcing()

    def train_epoch_teacher_forcing(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            _,L1 = labels.shape
            _,L2 = input_ids.shape
            L    = L1 + L2
            if batch_idx <= 3000000:
               full_ids  = torch.cat([input_ids, labels], dim=1)
               embeddings = self.embedding(full_ids)
               causal_mask = torch.tril(torch.ones((1, L, 1), device=embeddings.device, dtype=embeddings.dtype))
               embeddings = embeddings.masked_fill(causal_mask == 0, float('-inf'))
               outputs = self.model(embeddings,L)
               logits = self.output_layer(outputs)
               ce_loss = self.criterion(logits.view(-1, self.vocab_size), full_ids.view(-1))
            else:
               embeddings = self.embedding(input_ids)
               outputs = self.model(embeddings,L1)
               logits = self.output_layer(outputs)
               ce_loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
            probs = F.softmax(logits, dim=-1)
            labels_safe = labels.clone()
            labels_safe[labels == -100] = 0
            correct_probs = probs.gather(dim=-1, index=labels_safe.unsqueeze(-1)).squeeze(-1)
            valid_mask = (labels != -100).float()
            correct_probs = correct_probs * valid_mask
            rep_mask = (labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] != -100) & (labels[:, :-1] != -100)
            rep_penalty = -torch.log(1 - correct_probs[:, 1:] + 1e-8)
            rep_loss = (rep_mask.float() * rep_penalty).mean()
            lambda_rep = 0.0
            loss = ce_loss + lambda_rep * rep_loss

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            loss = loss / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps * input_ids.numel()
            total_tokens += input_ids.numel()
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.embedding.parameters()) + list(
                        self.output_layer.parameters()),
                    max_norm=self.config['grad_clip']
                )
                self.optimizer.step()
                self.scheduler.step(loss)
                self.optimizer.zero_grad()
            if batch_idx % self.config['log_interval'] == 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
                print(f'[TF] Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {avg_loss:.4f} | '
                      f'PPL: {ppl:.2f} ｜ '
                      #f'chayiLoss: {A_list_loss:.4f} | '
                      f'rep_loss: {lambda_rep * rep_loss:.4f}')
            if batch_idx % (self.config['log_interval'] * 100) == 0 and batch_idx > 0:
                print("\n测试文本生成...")
                prompt1 = "人工智能"
                generated4 = self.generate_text_new(prompt1, max_length=10, temperature=0.6, top_p=0.9)
                generated5 = self.generate_text_new(prompt1, max_length=10, temperature=0.8, top_p=0.9)
                generated6 = self.generate_text_new(prompt1, max_length=10, temperature=1.0, top_p=0.95)
                print(f"提示: {prompt1}\n生成: {generated4}")
                print(f"提示: {prompt1}\n生成: {generated5}")
                print(f"提示: {prompt1}\n生成: {generated6}")

        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.embedding.parameters()) + list(
                    self.output_layer.parameters()),
                max_norm=self.config['grad_clip']
            )
            self.optimizer.step()
            self.scheduler.step(loss)
            self.optimizer.zero_grad()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return avg_loss, ppl
    def train_hybrid(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if  batch_idx <= 500:
               input_ids = batch['input_ids'].to(self.device)  # (B, L)
               labels = batch['labels'].to(self.device)  # (B, L)
               attention_mask = batch['attention_mask'].to(self.device)

               embeddings = self.embedding(input_ids)  # (B, L, D)
               outputs = self.model(embeddings)  # (B, L, D)
               logits = self.output_layer(outputs)  # (B, L, vocab_size)

               loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))

               self.optimizer.zero_grad()
               loss.backward()
               torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.embedding.parameters()) + list(
                    self.output_layer.parameters()),
                    max_norm=self.config['grad_clip']
                )
               self.optimizer.step()
               self.scheduler.step()
               total_loss += loss.item() * input_ids.numel()
               total_tokens += input_ids.numel()
            else :
                if batch_idx % 2 == 0:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    B, L = input_ids.shape
                    h = None
                    total_loss = 0.0

                    for t in range(L):
                        x_t = input_ids[:, t]
                        x_emb = self.embedding(x_t)
                        if h is None:
                            y, h = self.model.step(x_emb)
                        else:
                            y, h = self.model.step(x_emb, h)
                        logits = self.output_layer(y)
                        target_t = labels[:, t]
                        step_loss = self.criterion(logits, target_t)
                        total_loss += step_loss

                    loss = total_loss / L
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.embedding.parameters()) + list(
                            self.output_layer.parameters()),
                        max_norm=self.config['grad_clip'])
                    self.optimizer.step()
                    self.scheduler.step()

                    total_loss += loss.item() * B * L
                    total_tokens += B * L
                else:
                    input_ids = batch['input_ids'].to(self.device)  # (B, L)
                    labels = batch['labels'].to(self.device)  # (B, L)
                    attention_mask = batch['attention_mask'].to(self.device)

                    embeddings = self.embedding(input_ids)  # (B, L, D)
                    outputs = self.model(embeddings)  # (B, L, D)
                    logits = self.output_layer(outputs)  # (B, L, vocab_size)

                    loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.embedding.parameters()) + list(
                            self.output_layer.parameters()),
                        max_norm=self.config['grad_clip']
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    total_loss += loss.item() * input_ids.numel()
                    total_tokens += input_ids.numel()
            if batch_idx % self.config['log_interval'] == 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
                print(f'[TF] Epoch: {self.current_epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {avg_loss:.4f} | PPL: {ppl:.2f}')

            if batch_idx % (self.config['log_interval'] * 100) == 0 and batch_idx > 0:
                print("\n测试文本生成...")
                prompt = "人工智能"
                generated = self.generate_text(prompt, max_length=10, temperature=0.6)
                print(f"提示: {prompt}\n生成: {generated}")
            #if batch_idx % 1000 == 0 and batch_idx > 0:
                #val_loss, val_ppl = self.validate_teacher_forcing()
                #print(f'[Val] Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}')

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
            'loss': loss,
            'config': self.config,
            'tokenizer': self.tokenizer,
        }

        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"新的最佳模型已保存: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        self.output_layer.load_state_dict(checkpoint['output_layer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
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
    def generate_text1(self, prompt, max_length=100, temperature=0.8):
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
    def generate_text4(self, prompt, max_length=100, temperature=1.0):
        """自回归生成文本"""
        self.model.eval()
        prompt = self.preprocess_text(prompt)
        tokens = [self.vocab.get(char, 0) for char in prompt]
        generated = tokens.copy()
        seq_len = self.config['seq_len']

        with torch.no_grad():
            for _ in range(max_length):
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
        'data_path': 'MD19950617/BelleGroup_train_3.5M_CN',
        'batch_size': 1,
        'seq_len': 128,
        'd_model': 1024,
        'd_state': 1024,
        'learning_rate': 5e-5,
        'weight_decay': 1e-2,
        'grad_clip': 1,
        'num_epochs': 1,
        'log_interval': 1,
        'save_interval': 5,
        'save_dir': 'checkpoints',
        'num_workers': 0,
        'num_blocks':16,
        'train_jsonl_path':'distill_r1_110k_sft.jsonl',
        'gradient_accumulation_steps':16,
    }
    print('初始化')
    # 创建训练器
    trainer = BaijuTrainer(config)
    print('初始化完成')
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被中断，保存检查点...")
        trainer.save_checkpoint(trainer.current_epoch, trainer.best_loss)

    # 测试文本生成
    print("\n测试文本生成...")
    prompt = "人工智能"
    generated = trainer.generate_text_new(prompt, max_length=50, temperature=0.4)
    print(f"提示: {prompt}")
    print(f"生成: {generated}")
def train_continue():
    config = {
        'data_path': 'MD19950617/BelleGroup_train_3.5M_CN',
        'batch_size': 1,
        'seq_len': 128,
        'd_model': 512,
        'd_state': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'grad_clip': 1,
        'num_epochs': 1,
        'log_interval': 1,
        'save_interval': 5,
        'save_dir': 'checkpoints',
        'num_workers': 0,
        'num_blocks':2,
        'train_jsonl_path': 'distill_r1_110k_sft.jsonl'
    }

    # 创建训练器
    trainer = BaijuTrainer(config)
    trainer.load_checkpoint('checkpoints/best_model.pth')
    print('初始化完成')
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被中断，保存检查点...")
        trainer.save_checkpoint(trainer.current_epoch, trainer.best_loss)

    # 测试文本生成
    print("\n测试文本生成...")
    prompt = "人工智能"
    generated = trainer.generate_text(prompt, max_length=50,temperature=0.4,top_k=40)
    print(f"提示: {prompt}")
    print(f"生成: {generated}")
def eval(prompt, max_length=100, temperature=1.0):
    config = {
        'data_path': 'MD19950617/BelleGroup_train_3.5M_CN',
        'batch_size': 4,
        'seq_len': 64,
        'd_model': 32,
        'd_state': 2048,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'grad_clip': 1,
        'num_epochs': 1,
        'log_interval': 1,
        'save_interval': 5,
        'save_dir': 'checkpoints',
        'num_workers': 0,
        'num_blocks':1,
        'train_jsonl_path':'distill_r1_110k_sft.jsonl'
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
    #prompt = "人工智能"
    #generated = BaijuTrainer.generate_text(prompt, max_length=50, temperature=0.6, top_k=50)
    #print(f"提示: {prompt}")
    #print(f"生成: {generated}")