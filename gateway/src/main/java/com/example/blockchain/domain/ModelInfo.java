package com.example.blockchain.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class ModelInfo {
    /** 模型类型 */
    private String model_type;

    /** 模型版本 */
    private String model_version;

    /** 加载时间 */
    private String loaded_at;

    /** 性能指标 */
    private Map<String, Object> performance_metrics;

    /** 特征数量 */
    private int num_features;

    /** 类别数量 */
    private int num_classes;

    /** 隐藏层维度 */
    private int hidden_channels;

    /** GNN层数 */
    private int gnn_layers;

    /** 随机森林树数量 */
    private int rf_n_estimators;

    /** 随机森林最大深度 */
    private int rf_max_depth;

    /** 实验名称 */
    private String experiment_name;

    /** 检查点目录 */
    private String checkpoint_dir;

    /** 模型状态 */
    private String status;

    /** 推理阈值 */
    private Float threshold;

    /** 缓存构建时间 */
    private String cache_built_at;
}