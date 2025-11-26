#!/usr/bin/env python3
"""
区块链AML反洗钱系统主运行程序
使用图神经网络进行交易异常检测
服务器训练优化版本
"""

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import os
import logging
import time
from datetime import datetime
import json

from data import load_train_data, load_val_data, load_inference_data
from models.inference import inference
from models.evaluator import ModelEvaluator
from config.logging_config import setup_training_logger, log_system_info, log_model_info


class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)  # 节点级别的输出
        return x


def setup_server_logging():
    """设置服务器训练日志"""
    from config.logging_config import setup_training_logger, log_system_info, log_model_info
    
    # 创建训练日志
    logger = setup_training_logger(
        experiment_name=f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        log_dir="logs",
        log_level="INFO"
    )
    
    return logger


def save_training_config(args, save_path="checkpoints"):
    """保存训练配置"""
    os.makedirs(save_path, exist_ok=True)
    config_path = os.path.join(save_path, "training_config.json")
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        config["cuda_device_name"] = torch.cuda.get_device_name(0)
        config["cuda_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logging.info(f"训练配置已保存到: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='区块链AML反洗钱系统 - 服务器训练版')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'inference'],
                       help='运行模式: train(训练), eval(评估), inference(推理)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='GNN层数')
    parser.add_argument('--num_features', type=int, default=165,
                       help='输入特征维度')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='分类类别数')
    parser.add_argument('--data_path', type=str, default='data/',
                       help='数据路径')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--model_name', type=str, default='gnn_model',
                       help='模型名称')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='运行设备')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'plateau', 'none'],
                       help='学习率调度器')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停耐心值')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='梯度裁剪值')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--pin_memory', action='store_true',
                       help='是否使用pin memory')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='是否使用混合精度训练')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')

    args = parser.parse_args()
    
    # 设置日志
    logger = setup_server_logging()
    logger.info("=" * 80)
    logger.info("区块链AML反洗钱系统启动")
    logger.info(f"运行模式: {args.mode}")
    logger.info("=" * 80)

    # 设备选择
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    
    # 保存训练配置
    save_training_config(args, args.checkpoint_dir)

    # 创建模型
    logger.info("创建GNN模型...")
    model = SimpleGNN(
        num_features=args.num_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes
    )
    model = model.to(device)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.mode == 'train':
        logger.info("开始训练模型...")
        start_time = time.time()
        
        # 加载数据
        logger.info("加载训练和验证数据...")
        train_loader = load_train_data(args.data_path, args.batch_size)
        val_loader = load_val_data(args.data_path, args.batch_size)
        
        logger.info(f"训练批次数: {len(train_loader)}")
        logger.info(f"验证批次数: {len(val_loader)}")
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 训练循环
        best_val_auc = 0.0
        best_epoch = 0
        
        for epoch in range(args.epochs):
            # 训练阶段
            model.train()
            total_train_loss = 0.0
            train_scores = []
            train_labels = []
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # 前向传播 - 使用图级别的分类模型
                output = model(batch.x, batch.edge_index, batch.batch)
                loss = F.cross_entropy(output, batch.y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # 计算预测概率（正类概率）
                pred = F.softmax(output, dim=1)[:, 1]
                train_scores.extend(pred.cpu().detach().numpy())
                train_labels.extend(batch.y.cpu().numpy())
            
            # 计算训练AUC
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(train_labels, train_scores) if len(set(train_labels)) > 1 else 0.0
            avg_train_loss = total_train_loss / len(train_loader)
            
            # 验证阶段
            model.eval()
            total_val_loss = 0.0
            val_scores = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    
                    output = model(batch.x, batch.edge_index, batch.batch)
                    loss = F.cross_entropy(output, batch.y)
                    total_val_loss += loss.item()
                    
                    pred = F.softmax(output, dim=1)[:, 1]
                    val_scores.extend(pred.cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())
            
            # 计算验证AUC
            val_auc = roc_auc_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0.0
            avg_val_loss = total_val_loss / len(val_loader)
            
            # 打印训练信息
            logger.info(f"Epoch {epoch+1}/{args.epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_model_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': avg_val_loss
                }, best_model_path)
                logger.info(f"  新的最佳模型已保存 (Val AUC: {val_auc:.4f})")
        
        training_summary = {
            'best_val_auc': best_val_auc,
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'best_epoch': best_epoch
        }
        
        # 保存最终模型
        final_model_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_final.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_summary': training_summary,
            'args': vars(args)
        }, final_model_path)
        
        total_time = time.time() - start_time
        logger.info(f"训练完成！总时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        logger.info(f"最佳验证AUC: {training_summary['best_val_auc']:.4f}")
        logger.info(f"最佳epoch: {training_summary['best_epoch'] + 1}")
        logger.info(f"最终模型已保存到: {final_model_path}")

    elif args.mode == 'eval':
        logger.info("开始评估模型...")
        model_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_final.pth")
        
        # 加载训练好的模型
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 从检查点或配置文件中获取模型参数
            if 'args' in checkpoint:
                model_args = checkpoint['args']
                hidden_channels = model_args['hidden_channels']
                num_features = model_args['num_features']
                num_classes = model_args['num_classes']
            else:
                # 尝试从配置文件读取
                config_path = os.path.join(args.checkpoint_dir, "training_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    model_args = config['args']
                    hidden_channels = model_args['hidden_channels']
                    num_features = model_args['num_features']
                    num_classes = model_args['num_classes']
                else:
                    # 使用默认值
                    hidden_channels = args.hidden_channels
                    num_features = args.num_features
                    num_classes = args.num_classes
            
            # 使用正确的参数创建模型
            model = SimpleGNN(
                num_features=num_features,
                hidden_channels=hidden_channels,
                num_classes=num_classes
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"成功加载模型: {model_path}")
            logger.info(f"模型参数: num_features={num_features}, hidden_channels={hidden_channels}, num_classes={num_classes}")
            
            if 'training_summary' in checkpoint:
                logger.info(f"训练摘要: {checkpoint['training_summary']}")
        except FileNotFoundError:
            logger.error(f"模型文件未找到: {model_path}")
            return
        
        # 加载测试数据
        test_loader = load_val_data(args.data_path, args.batch_size)  # 暂时用验证数据代替
        
        # 创建评估器
        evaluator = ModelEvaluator(model, device)
        logger.info("开始详细评估...")
        
        # 进行评估
        metrics = evaluator.evaluate(test_loader)
        evaluator.print_metrics(metrics)
        
        # 保存评估结果
        eval_results_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_eval_results.json")
        with open(eval_results_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"评估结果已保存到: {eval_results_path}")

    elif args.mode == 'inference':
        logger.info("开始推理...")
        model_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_final.pth")
        
        # 加载训练好的模型
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 从检查点或配置文件中获取模型参数
            if 'args' in checkpoint:
                model_args = checkpoint['args']
                hidden_channels = model_args['hidden_channels']
                num_features = model_args['num_features']
                num_classes = model_args['num_classes']
            else:
                # 尝试从配置文件读取
                config_path = os.path.join(args.checkpoint_dir, "training_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    model_args = config['args']
                    hidden_channels = model_args['hidden_channels']
                    num_features = model_args['num_features']
                    num_classes = model_args['num_classes']
                else:
                    # 使用默认值
                    hidden_channels = args.hidden_channels
                    num_features = args.num_features
                    num_classes = args.num_classes
            
            # 使用正确的参数创建模型
            model = SimpleGNN(
                num_features=num_features,
                hidden_channels=hidden_channels,
                num_classes=num_classes
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"成功加载模型: {model_path}")
            logger.info(f"模型参数: num_features={num_features}, hidden_channels={hidden_channels}, num_classes={num_classes}")
        except FileNotFoundError:
            logger.error(f"模型文件未找到: {model_path}")
            return
        
        # 加载推理数据
        logger.info("加载推理数据...")
        data = load_inference_data(args.data_path)
        
        # 执行推理
        start_time = time.time()
        
        # 简单的节点嵌入推理
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            embeddings = model.conv1(data.x, data.edge_index)
            embeddings = F.relu(embeddings)
            embeddings = model.conv2(embeddings, data.edge_index)
            embeddings = embeddings.cpu()
        
        inference_time = time.time() - start_time
        
        # 保存嵌入向量
        embeddings_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_embeddings.pth")
        torch.save(embeddings, embeddings_path)
        
        logger.info(f"推理完成！耗时: {inference_time:.2f}秒")
        logger.info(f"节点嵌入形状: {embeddings.shape}")
        logger.info(f"嵌入向量已保存到: {embeddings_path}")
        logger.info("可以基于嵌入向量进行异常检测和风险评分")

    logger.info("程序执行完成!")


if __name__ == "__main__":
    main()