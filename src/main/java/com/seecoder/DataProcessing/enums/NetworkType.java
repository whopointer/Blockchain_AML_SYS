package com.seecoder.DataProcessing.enums;

/**
 * 区块链网络类型枚举
 * 作用：定义支持的区块链网络及其对应的 BigQuery 数据集名称
 * 设计模式：枚举模式，用于定义一组固定的常量
 *
 * 每个枚举值包含两部分：
 * 1. 枚举常量名（如 ETHEREUM_MAINNET）
 * 2. 对应的 BigQuery 数据集名称（如 "ethereum-mainnet"）
 */
public enum NetworkType {
    /**
     * 以太坊主网
     * 对应 BigQuery 数据集：crypto_ethereum
     */
    ETHEREUM_MAINNET("ethereum-mainnet"),

    /**
     * 以太坊 Goerli 测试网
     * 对应 BigQuery 数据集：crypto_ethereum_goerli
     */
    ETHEREUM_GOERLI("ethereum-goerli"),

    /**
     * BSC（币安智能链）主网
     * 对应 BigQuery 数据集：crypto_bsc
     */
    BSC_MAINNET("bsc-mainnet"),

    /**
     * Polygon（原 Matic）主网
     * 对应 BigQuery 数据集：crypto_polygon
     */
    POLYGON_MAINNET("polygon-mainnet"),

    /**
     * Arbitrum 主网
     * 对应 BigQuery 数据集：crypto_arbitrum
     */
    ARBITRUM_MAINNET("arbitrum-mainnet");

    /**
     * 私有字段：BigQuery 数据集名称
     * final 修饰表示一旦赋值不可修改
     */
    private final String datasetName;

    /**
     * 枚举构造函数
     * 注意：枚举的构造函数默认是私有的
     *
     * @param datasetName BigQuery 数据集名称
     */
    NetworkType(String datasetName) {
        this.datasetName = datasetName;
    }

    /**
     * 获取数据集名称的公共方法
     * 用于在构建 BigQuery SQL 查询时获取正确的数据集名称
     *
     * @return 对应区块链的 BigQuery 数据集名称
     */
    public String getDatasetName() {
        return datasetName;
    }

    /**
     * 示例：在查询中的使用方式
     * String query = "SELECT * FROM `bigquery-public-data.crypto_" + network.getDatasetName() + ".transactions`";
     *
     * 对于 ETHEREUM_MAINNET，会生成：
     * SELECT * FROM `bigquery-public-data.crypto_ethereum-mainnet.transactions`
     */
}