import torch
import torch.nn as nn
from data_loader import create_data_loader
from mamba import MAMBA

def test_training_flow():
    print("=== 测试训练流程 ===")
    
    # 创建数据加载器
    print("1. 创建数据加载器...")
    dataloader, vocab, inv_vocab = create_data_loader(
        '都市极品医神.txt', 
        batch_size=4, 
        seq_len=128
    )
    print(f"   词汇表大小: {len(vocab)}")
    print(f"   数据加载器长度: {len(dataloader)}")
    
    # 创建模型和组件
    print("2. 创建模型组件...")
    d_model = 128
    d_state = 32
    
    model = MAMBA(d_model=d_model, d_state=d_state)
    embedding = nn.Embedding(len(vocab), d_model)
    output_layer = nn.Linear(d_model, len(vocab))
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(embedding.parameters()) + list(output_layer.parameters()),
        lr=1e-3
    )
    
    print("3. 测试一个训练步骤...")
    
    # 获取一个batch
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"   Batch {batch_idx}:")
        print(f"     输入形状: {inputs.shape}")
        print(f"     目标形状: {targets.shape}")
        
        # 前向传播
        embeddings = embedding(inputs)
        outputs = model(embeddings)
        logits = output_layer(outputs)
        
        print(f"     嵌入形状: {embeddings.shape}")
        print(f"     模型输出形状: {outputs.shape}")
        print(f"     逻辑输出形状: {logits.shape}")
        
        # 计算损失
        loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(embedding.parameters()) + list(output_layer.parameters()),
            max_norm=1.0
        )
        
        # 优化步骤
        optimizer.step()
        
        print(f"     损失值: {loss.item():.4f}")
        print("   ✓ 训练步骤完成")
        
        # 只测试一个batch
        break
    
    print("4. 测试模型保存和加载...")
    
    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'embedding_state_dict': embedding.state_dict(),
        'output_layer_state_dict': output_layer.state_dict(),
        'vocab': vocab,
        'config': {'d_model': d_model, 'd_state': d_state}
    }
    
    torch.save(checkpoint, 'test_checkpoint.pth')
    print("   ✓ 模型保存成功")
    
    # 加载模型
    loaded_checkpoint = torch.load('test_checkpoint.pth')
    
    # 创建新模型
    new_model = MAMBA(d_model=d_model, d_state=d_state)
    new_embedding = nn.Embedding(len(vocab), d_model)
    new_output_layer = nn.Linear(d_model, len(vocab))
    
    # 加载状态
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    new_embedding.load_state_dict(loaded_checkpoint['embedding_state_dict'])
    new_output_layer.load_state_dict(loaded_checkpoint['output_layer_state_dict'])
    
    print("   ✓ 模型加载成功")
    
    # 测试加载的模型
    with torch.no_grad():
        test_inputs = inputs[:2]  # 取前两个样本
        test_embeddings = new_embedding(test_inputs)
        test_outputs = new_model(test_embeddings)
        test_logits = new_output_layer(test_outputs)
        
        print(f"     测试输入形状: {test_inputs.shape}")
        print(f"     测试输出形状: {test_outputs.shape}")
        print("   ✓ 加载模型测试通过")
    
    # 清理测试文件
    import os
    if os.path.exists('test_checkpoint.pth'):
        os.remove('test_checkpoint.pth')
        print("   ✓ 测试文件清理完成")
    
    print("\n=== 所有测试通过！ ===")
    print("训练流程可以正常运行，维度匹配正确。")
    print("现在可以运行 train.py 开始正式训练。")

if __name__ == "__main__":
    test_training_flow()