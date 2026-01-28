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
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
import os
import logging
import time
from datetime import datetime
import json
import numpy as np

from data import EllipticDataset, load_inference_data, load_val_data
from models.two_stage_dgi_rf import create_two_stage_dgi_rf
from config.logging_config import setup_training_logger, log_system_info, log_model_info,setup_evaluation_logger, setup_inference_logger


def setup_server_logging(mode: str):
    """设置按模式分类的服务器日志"""
    
    # 根据模式选择合适的日志设置函数
    logger_setup_map = {
        'gnn_dgi_rf': setup_training_logger,
        'training': setup_training_logger,
        'eval': setup_evaluation_logger,
        'evaluation': setup_evaluation_logger,
        'inference': setup_inference_logger
    }
    
    if mode not in logger_setup_map:
        raise ValueError(f"不支持的运行模式: {mode}")
    
    # 使用对应的日志设置函数
    setup_func = logger_setup_map[mode]
    logger = setup_func(
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


def build_time_split_masks(time_steps: torch.Tensor, test_size: float = 0.2, val_size: float = 0.2):
    """按时间步划分 train/val/test mask（单大图训练用）"""
    unique_steps = sorted(time_steps.unique().cpu().tolist())
    n_steps = len(unique_steps)
    if n_steps < 3:
        train_steps = unique_steps[:1]
        val_steps = unique_steps[1:2] if len(unique_steps) > 1 else []
        test_steps = unique_steps[2:] if len(unique_steps) > 2 else []
    else:
        test_start = max(1, int(n_steps * (1 - test_size)))
        val_start = max(1, int(test_start * (1 - val_size / (1 - test_size))))
        train_steps = unique_steps[:val_start]
        val_steps = unique_steps[val_start:test_start]
        test_steps = unique_steps[test_start:]

    steps_tensor = time_steps
    train_mask = torch.isin(steps_tensor, torch.tensor(train_steps, dtype=steps_tensor.dtype))
    val_mask = torch.isin(steps_tensor, torch.tensor(val_steps, dtype=steps_tensor.dtype))
    test_mask = torch.isin(steps_tensor, torch.tensor(test_steps, dtype=steps_tensor.dtype))
    return train_mask, val_mask, test_mask, train_steps, val_steps, test_steps


def log_multi_threshold_metrics(logger, y_true, y_proba_pos, thresholds):
    """输出多阈值下的精确率/召回率/F1，便于校准阈值"""
    from sklearn.metrics import precision_score, recall_score, f1_score

    lines = []
    for thr in thresholds:
        y_pred = (y_proba_pos >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        lines.append(f"thr={thr:.2f} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

    logger.info("多阈值对比:")
    for line in lines:
        logger.info(f"  {line}")

def find_optimal_threshold_from_proba(y_true, y_proba_pos, metric="f1", recall_min=None, thresholds=None):
    """基于已计算的正类概率搜索最优阈值（用于评估阶段校准）"""
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_proba_pos = np.asarray(y_proba_pos).reshape(-1)

    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.02)

    best_score = -1.0
    best_thr = 0.5
    best_recall = -1.0

    for thr in thresholds:
        y_pred = (y_proba_pos >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"不支持的指标: {metric}")

        rec = recall_score(y_true, y_pred, zero_division=0)
        if recall_min is not None and rec < recall_min:
            continue

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_recall = rec

    if best_score < 0 and recall_min is not None:
        for thr in thresholds:
            y_pred = (y_proba_pos >= thr).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"不支持的指标: {metric}")
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_recall = recall_score(y_true, y_pred, zero_division=0)

    return best_thr, best_score, best_recall


def get_pos_proba_from_probs(model, probs):
    """根据 RF 的 classes_ 稳健取出正类(1)概率"""
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs
    try:
        classes = model.rf_classifier.classifier.classes_
        pos_idx = int(np.where(classes == 1)[0][0])
    except Exception:
        pos_idx = 1 if probs.shape[1] > 1 else 0
    return probs[:, pos_idx]

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
    
    # 设置按模式分类的日志
    logger = setup_server_logging(args.mode)
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
            rf_class_weight="balanced_subsample",
            rf_auto_class_weight=False,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment_name,
            balance_strategy='none',         # 关闭欠采样，避免与 class_weight 叠加过度偏正类
            loss_type='balanced_focal'       # 使用平衡Focal Loss
        )
        
        logger.info(f"GIN编码器参数数量: {sum(p.numel() for p in model.dgi_model.gin_encoder.parameters()):,}")
        
        # 加载单大图数据
        logger.info("加载单大图数据...")
        dataset = EllipticDataset(root=args.data_path, include_unknown=True)
        raw_labels = dataset.merged_df["class"].astype(str).values
        label_map = {"1": 1, "2": 2}
        y_raw = np.array([label_map.get(v, -1) for v in raw_labels], dtype=np.int64)
        data = Data(
            x=dataset.x,
            edge_index=dataset.edge_index,
            y=torch.tensor(y_raw, dtype=torch.long),
            time_steps=dataset.time_steps,
            num_nodes=dataset.x.shape[0]
        )
        data.train_mask, data.val_mask, data.test_mask, train_steps, val_steps, test_steps = build_time_split_masks(
            data.time_steps
        )

        logger.info(f"时间步分割: 训练({len(train_steps)}步), 验证({len(val_steps)}步), 测试({len(test_steps)}步)")
        logger.info(
            f"节点数量: train={int(data.train_mask.sum())}, val={int(data.val_mask.sum())}, "
            f"test={int(data.test_mask.sum())}"
        )
        
        # 开始两阶段训练
        start_time = time.time()
        
        try:
            training_results = model.end_to_end_train_single_graph(
                data=data,
                dgi_epochs=args.dgi_epochs,
                rf_hyperparameter_tuning=args.rf_hyperparameter_tuning,
                learning_rate=args.lr,
                patience=args.patience,
                threshold_metric="f1",
                include_unknown_in_threshold=False,
                threshold_recall_min=0.5
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
            threshold = None
            if isinstance(training_results, dict):
                threshold = (
                    training_results.get("stage2", {}).get("optimal_threshold")
                    if training_results.get("stage2")
                    else None
                )
            all_predictions, all_probabilities = model.predict_single_graph(
                data,
                mask=None,
                threshold=threshold
            )

            val_idx = torch.nonzero(data.val_mask, as_tuple=False).view(-1).cpu().numpy()
            val_labels_raw = data.y[val_idx].cpu().numpy()
            known_mask = val_labels_raw >= 0
            val_labels_raw = val_labels_raw[known_mask]
            val_labels = (val_labels_raw == 1).astype(int)
            val_predictions_raw = all_predictions[val_idx][known_mask]
            val_predictions = (val_predictions_raw == 1).astype(int)
            val_probabilities = all_probabilities[val_idx][known_mask]
            
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

            # 多阈值对比（使用正类概率列）
            log_multi_threshold_metrics(
                logger,
                val_labels,
                val_probabilities[:, 1],
                thresholds=[0.3, 0.5, 0.7, 0.9]
            )
            
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
            rf_class_weight="balanced_subsample",
            rf_auto_class_weight=False,
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
        
        # 加载单大图数据
        dataset = EllipticDataset(root=args.data_path, include_unknown=False)
        raw_labels = dataset.merged_df["class"].astype(str).values
        label_map = {"1": 1, "2": 2}
        y_raw = np.array([label_map.get(v, -1) for v in raw_labels], dtype=np.int64)
        data = Data(
            x=dataset.x,
            edge_index=dataset.edge_index,
            y=torch.tensor(y_raw, dtype=torch.long),
            time_steps=dataset.time_steps,
            num_nodes=dataset.x.shape[0]
        )
        data.train_mask, data.val_mask, data.test_mask, _, _, _ = build_time_split_masks(data.time_steps)
        
        # 进行预测和评估
        logger.info("开始详细评估...")
        
        # 获取评估集数据
        eval_mask = data.test_mask if int(data.test_mask.sum()) > 0 else data.val_mask
        eval_idx = torch.nonzero(eval_mask, as_tuple=False).view(-1).cpu().numpy()
        true_labels_raw = data.y[eval_idx].cpu().numpy()
        known_mask = true_labels_raw >= 0
        true_labels_raw = true_labels_raw[known_mask]
        true_labels = (true_labels_raw == 1).astype(int)
        
        # 直接进行预测（不指定阈值，使用模型默认的predict_proba）
        all_predictions, all_probabilities = model.predict_single_graph(data, mask=None)
        predictions_raw = all_predictions[eval_idx][known_mask]
        predictions = (predictions_raw == 1).astype(int)
        probabilities = all_probabilities[eval_idx][known_mask]

        # 评估阶段重新校准阈值（基于当前 eval 集的概率分布）
        y_proba_pos = get_pos_proba_from_probs(model, probabilities)
        recal_thr, recal_score, recal_recall = find_optimal_threshold_from_proba(
            true_labels,
            y_proba_pos,
            metric="f1",
            recall_min=0.5
        )
        logger.info(
            f"评估阶段校准阈值: {recal_thr:.3f} (metric=f1, score={recal_score:.4f}, recall={recal_recall:.4f})"
        )

        # 用校准阈值重新计算预测
        predictions = (y_proba_pos >= recal_thr).astype(int)

        # 计算评估指标
        from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

        auc = roc_auc_score(true_labels, y_proba_pos) if len(set(true_labels)) > 1 else 0.0
        ap = average_precision_score(true_labels, y_proba_pos) if len(set(true_labels)) > 1 else 0.0
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

        # 多阈值对比（使用正类概率列）
        log_multi_threshold_metrics(
            logger,
            true_labels,
            y_proba_pos,
            thresholds=[0.3, 0.5, 0.7, 0.9]
        )
        
        # 保存评估结果
        eval_results = {
            'auc': float(auc),
            'average_precision': float(ap),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': report,
            'calibrated_threshold': float(recal_thr),
            'calibration': {
                'metric': 'f1',
                'recall_min': 0.5,
                'score': float(recal_score),
                'recall': float(recal_recall)
            }
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
        
        # 加载单大图推理数据
        logger.info("加载推理数据...")
        dataset = EllipticDataset(root=args.data_path, include_unknown=False)
        raw_labels = dataset.merged_df["class"].astype(str).values
        label_map = {"1": 1, "2": 2}
        y_raw = np.array([label_map.get(v, -1) for v in raw_labels], dtype=np.int64)
        data = Data(
            x=dataset.x,
            edge_index=dataset.edge_index,
            y=torch.tensor(y_raw, dtype=torch.long),
            time_steps=dataset.time_steps,
            num_nodes=dataset.x.shape[0]
        )
        data.train_mask, data.val_mask, data.test_mask, _, _, _ = build_time_split_masks(data.time_steps)
        
        # 执行推理
        start_time = time.time()
        
        try:
            # 优先使用评估阶段保存的校准阈值
            optimal_threshold = None
            threshold_source = "unknown"
            
            # 首先尝试从评估结果文件中读取校准阈值
            try:
                eval_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_eval_results.json")
                if os.path.exists(eval_path):
                    with open(eval_path, "r", encoding="utf-8") as f:
                        eval_results = json.load(f)
                    calibrated_threshold = eval_results.get("calibrated_threshold")
                    if calibrated_threshold is not None:
                        optimal_threshold = float(calibrated_threshold)
                        if 0.0 < optimal_threshold < 1.0:
                            threshold_source = "eval_calibrated"
                            logger.info(f"使用评估阶段校准阈值: {optimal_threshold:.3f}")
                        else:
                            logger.warning(f"评估阈值 {optimal_threshold:.3f} 无效，回退到训练阈值")
                            optimal_threshold = None
            except Exception as e:
                logger.warning(f"读取评估结果文件失败: {e}，回退到训练阈值")
            
            # 如果没有评估阈值，使用训练阶段的最优阈值
            if optimal_threshold is None:
                if hasattr(model, "training_history") and isinstance(model.training_history, dict):
                    optimal_threshold = (
                        model.training_history.get("stage2", {}).get("optimal_threshold")
                    )
                if optimal_threshold is not None:
                    threshold_source = "training_optimal"
                    logger.info(f"使用训练阶段最优阈值: {optimal_threshold:.3f}")
            
            # 最后使用默认阈值作为兜底
            if optimal_threshold is None:
                optimal_threshold = 0.5
                threshold_source = "default"
                logger.warning("未找到任何有效阈值，使用默认阈值 0.5")
            
            logger.info(f"推理阈值来源={threshold_source}, value={optimal_threshold:.3f}")
            
            # 进行预测（使用更低的阈值以允许更多正常交易被识别）
            predictions, probabilities = model.predict_single_graph(data, threshold=optimal_threshold)
            
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
            logger.info(f"正常交易预测数量: {np.sum(predictions == 2)}")
            logger.info(f"平均异常概率: {np.mean(probabilities[:, 1]):.4f}")
            logger.info(f"预测结果已保存到: {results_path}")
            
        except Exception as e:
            logger.error(f"推理过程中出现错误: {e}")
            return

    

    logger.info("程序执行完成!")


if __name__ == "__main__":
    main()
