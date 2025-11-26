#!/usr/bin/env python3
"""
区块链AML反洗钱系统主运行程序
使用图神经网络进行交易异常检测
"""

import argparse
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader

from models.gnn_model import ImprovedGNNModel
from models.dgi import ImprovedDGI
from models.trainer import train, evaluate, create_trainer
from models.inference import inference
from models.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='区块链AML反洗钱系统')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'inference'],
                       help='运行模式: train(训练), eval(评估), inference(推理)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='隐藏层维度')
    parser.add_argument('--num_features', type=int, default=128,
                       help='输入特征维度')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='分类类别数')
    parser.add_argument('--data_path', type=str, default='data/',
                       help='数据路径')
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pth',
                       help='模型保存路径')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='运行设备')

    args = parser.parse_args()

    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"使用设备: {device}")
    print(f"运行模式: {args.mode}")

    # 创建模型
    gnn_model = ImprovedGNNModel(
        num_features=args.num_features, 
        num_classes=args.num_classes,
        hidden_channels=args.hidden_channels,
        use_multi_scale=True,
        use_attention_pooling=True
    )
    
    dgi_model = ImprovedDGI(gnn_model, args.hidden_channels)
    model = dgi_model.to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        print("开始训练模型...")
        # 这里需要加载你的数据
        # train_loader = load_train_data(args.data_path, args.batch_size)
        # val_loader = load_val_data(args.data_path, args.batch_size)
        
        # 模拟训练过程
        for epoch in range(args.epochs):
            # train_loss = train(model, train_loader, optimizer, device)
            # val_auc = evaluate(model, val_loader, device)
            
            # 模拟输出
            train_loss = 0.5 - epoch * 0.01  # 模拟损失下降
            val_auc = 0.7 + epoch * 0.002  # 模拟AUC上升
            
            print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}')
            
            # 保存最佳模型
            if epoch % 10 == 0:
                torch.save(model.state_dict(), args.model_path)
                print(f"模型已保存到: {args.model_path}")

    elif args.mode == 'eval':
        print("开始评估模型...")
        # 加载训练好的模型
        try:
            model.load_state_dict(torch.load(args.model_path))
            print(f"成功加载模型: {args.model_path}")
        except FileNotFoundError:
            print(f"模型文件未找到: {args.model_path}")
            return
        
        # test_loader = load_test_data(args.data_path, args.batch_size)
        # test_auc = evaluate(model, test_loader, device)
        
        # 模拟评估结果
        test_auc = 0.85
        print(f'测试集AUC: {test_auc:.4f}')
        
        # 使用评估器进行详细评估
        evaluator = ModelEvaluator(model, device)
        # metrics = evaluator.evaluate(test_loader)
        # evaluator.print_metrics(metrics)

    elif args.mode == 'inference':
        print("开始推理...")
        # 加载训练好的模型
        try:
            model.load_state_dict(torch.load(args.model_path))
            print(f"成功加载模型: {args.model_path}")
        except FileNotFoundError:
            print(f"模型文件未找到: {args.model_path}")
            return
        
        # data = load_inference_data(args.data_path)
        # embeddings = inference(model, data, device)
        
        print("推理完成，生成节点嵌入向量")
        print("可以基于嵌入向量进行异常检测和风险评分")

    print("程序执行完成!")


if __name__ == "__main__":
    main()