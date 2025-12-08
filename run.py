#!/usr/bin/env python3
"""
区块链AML反洗钱系统主运行程序
使用图神经网络进行交易异常检测
支持GNN+DGI+随机森林联合训练模式
"""

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
import os
import logging
import time
from datetime import datetime
import json
import numpy as np

from data import load_train_data, load_val_data, load_inference_data
from models.two_stage_dgi_rf import create_two_stage_dgi_rf
from config.logging_config import setup_training_logger, log_system_info, log_model_info


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
    parser.add_argument('--mode', type=str, default='gnn_dgi_rf', 
                       choices=['eval', 'inference', 'gnn_dgi_rf'],
                       help='运行模式: eval(评估), inference(推理), gnn_dgi_rf(GNN+DGI+随机森林联合训练)')
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
    
    # GNN+DGI+随机森林联合训练参数
    parser.add_argument('--dgi_epochs', type=int, default=100,
                       help='DGI预训练轮数')
    parser.add_argument('--rf_hyperparameter_tuning', action='store_true',
                       help='是否启用随机森林超参数调优')
    parser.add_argument('--experiment_name', type=str, default='gnn_dgi_rf_experiment',
                       help='实验名称')

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

    if args.mode == 'gnn_dgi_rf':
        logger.info("开始两阶段DGI+GIN+随机森林训练...")
        
        # 创建两阶段模型
        logger.info("创建两阶段DGI+GIN+随机森林模型...")
        model = create_two_stage_dgi_rf(
            num_features=args.num_features,
            num_classes=args.num_classes,
            hidden_channels=args.hidden_channels,
            gnn_layers=args.num_layers,
            rf_n_estimators=200,
            rf_max_depth=15,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment_name,
            balance_strategy='undersample',  # 使用欠采样策略
            loss_type='balanced_focal'       # 使用平衡Focal Loss
        )
        
        logger.info(f"GIN编码器参数数量: {sum(p.numel() for p in model.dgi_model.gin_encoder.parameters()):,}")
        
        # 加载数据
        logger.info("加载训练和验证数据...")
        train_loader = load_train_data(args.data_path, args.batch_size, args.num_workers)
        val_loader = load_val_data(args.data_path, args.batch_size, args.num_workers)
        
        logger.info(f"训练批次数: {len(train_loader)}")
        logger.info(f"验证批次数: {len(val_loader)}")
        
        # 开始两阶段训练
        start_time = time.time()
        
        try:
            training_results = model.end_to_end_train(
                train_loader=train_loader,
                val_loader=val_loader,
                dgi_epochs=args.dgi_epochs,
                rf_hyperparameter_tuning=args.rf_hyperparameter_tuning,
                learning_rate=args.lr,
                patience=args.patience
            )
            
            total_time = time.time() - start_time
            logger.info(f"两阶段训练完成！总时间: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
            
            # 保存训练摘要
            summary_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(training_results, f, indent=2, ensure_ascii=False)
            logger.info(f"训练摘要已保存到: {summary_path}")
            
            # 在验证集上进行最终评估
            logger.info("在验证集上进行最终评估...")
            val_predictions, val_probabilities = model.predict(val_loader)
            
            # 收集验证集真实标签
            val_labels = []
            for batch in val_loader:
                val_labels.extend(batch.y.numpy())
            val_labels = np.array(val_labels)
            
            # 计算最终指标
            from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
            final_auc = roc_auc_score(val_labels, val_probabilities[:, 1]) if len(set(val_labels)) > 1 else 0.0
            final_ap = average_precision_score(val_labels, val_probabilities[:, 1]) if len(set(val_labels)) > 1 else 0.0
            
            logger.info(f"最终验证AUC: {final_auc:.4f}")
            logger.info(f"最终验证AP: {final_ap:.4f}")
            
            # 打印分类报告
            logger.info("最终分类报告:")
            report = classification_report(val_labels, val_predictions, 
                                        target_names=['正常(0)', '异常(1)'], zero_division=0)
            logger.info(f"\n{report}")
            
            # 保存最终结果
            final_results = {
                'training_results': training_results,
                'final_evaluation': {
                    'val_auc': final_auc,
                    'val_ap': final_ap,
                    'classification_report': report
                },
                'training_time': total_time,
                'experiment_config': vars(args)
            }
            
            final_results_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_final_results.json")
            with open(final_results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"最终结果已保存到: {final_results_path}")
            
        except Exception as e:
            logger.error(f"两阶段训练过程中出现错误: {e}")
            logger.error("请检查数据格式和模型配置")
            raise e

    elif args.mode == 'eval':
        logger.info("开始评估两阶段DGI+GIN+随机森林模型...")
        
        # 创建两阶段模型
        model = create_two_stage_dgi_rf(
            num_features=args.num_features,
            num_classes=args.num_classes,
            hidden_channels=args.hidden_channels,
            gnn_layers=args.num_layers,
            rf_n_estimators=200,
            rf_max_depth=15,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment_name
        )
        
        # 加载训练好的模型
        try:
            model.load_full_model(args.experiment_name)
            logger.info(f"成功加载两阶段DGI+GIN+RF模型")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return
        
        # 加载测试数据
        test_loader = load_val_data(args.data_path, args.batch_size, args.num_workers)  # 暂时用验证数据代替
        
        # 进行预测和评估
        logger.info("开始详细评估...")
        
        # 首先寻找最优阈值
        logger.info("在验证集上寻找最优分类阈值...")
        optimal_threshold = model.find_optimal_threshold(test_loader, metric='f1')
        logger.info(f"使用最优阈值: {optimal_threshold:.3f}")
        
        # 使用最优阈值进行预测
        predictions, probabilities = model.predict(test_loader, threshold=optimal_threshold)
        
        # 收集真实标签
        true_labels = []
        for batch in test_loader:
            true_labels.extend(batch.y.numpy())
        true_labels = np.array(true_labels)
        
        # 计算评估指标
        from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
        
        auc = roc_auc_score(true_labels, probabilities[:, 1]) if len(set(true_labels)) > 1 else 0.0
        ap = average_precision_score(true_labels, probabilities[:, 1]) if len(set(true_labels)) > 1 else 0.0
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # 打印评估结果
        logger.info(f"评估结果:")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  Average Precision: {ap:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        # 打印分类报告
        report = classification_report(true_labels, predictions, 
                                    target_names=['正常(0)', '异常(1)'], zero_division=0)
        logger.info(f"\n分类报告:\n{report}")
        
        # 保存评估结果
        eval_results = {
            'auc': float(auc),
            'average_precision': float(ap),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': report
        }
        
        eval_results_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_eval_results.json")
        with open(eval_results_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        logger.info(f"评估结果已保存到: {eval_results_path}")

    elif args.mode == 'inference':
        logger.info("开始两阶段DGI+GIN+随机森林推理...")
        
        # 创建两阶段模型
        model = create_two_stage_dgi_rf(
            num_features=args.num_features,
            num_classes=args.num_classes,
            hidden_channels=args.hidden_channels,
            gnn_layers=args.num_layers,
            rf_n_estimators=200,
            rf_max_depth=15,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment_name
        )
        
        # 加载训练好的模型
        try:
            model.load_full_model(args.experiment_name)
            logger.info(f"成功加载两阶段DGI+GIN+RF模型")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return
        
        # 加载推理数据（只包含有标签的数据）
        logger.info("加载推理数据...")
        data = load_inference_data(args.data_path, include_unknown=False, all_timesteps=True)
        
        # 创建DataLoader
        from torch_geometric.loader import DataLoader
        data_loader = DataLoader([data], batch_size=1, shuffle=False)
        
        # 执行推理
        start_time = time.time()
        
        try:
            # 首先寻找最优阈值（使用验证集）
            logger.info("寻找最优分类阈值...")
            val_loader = load_val_data(args.data_path, args.batch_size, args.num_workers)
            optimal_threshold = model.find_optimal_threshold(val_loader, metric='f1')
            logger.info(f"使用最优阈值: {optimal_threshold:.3f}")
            
            # 进行预测（使用更低的阈值以允许更多正常交易被识别）
            predictions, probabilities = model.predict(data_loader, threshold=0.3)
            
            inference_time = time.time() - start_time
            
            # 保存预测结果
            results = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'inference_time': inference_time,
                'num_samples': len(predictions)
            }
            
            results_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_inference_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"推理完成！耗时: {inference_time:.2f}秒")
            logger.info(f"预测样本数: {len(predictions)}")
            logger.info(f"异常交易预测数量: {np.sum(predictions == 1)}")
            logger.info(f"正常交易预测数量: {np.sum(predictions == 0)}")
            logger.info(f"平均异常概率: {np.mean(probabilities[:, 1]):.4f}")
            logger.info(f"预测结果已保存到: {results_path}")
            
        except Exception as e:
            logger.error(f"推理过程中出现错误: {e}")
            return

    

    logger.info("程序执行完成!")


if __name__ == "__main__":
    main()