package com.seecoder.DataProcessing.po;

import lombok.Data;
import java.math.BigInteger;
import java.util.Date;

/**
 * 区块数据实体类（Plain Object）
 * 作用：表示区块链中的一个区块信息
 * 设计模式：POJO（Plain Old Java Object）模式，用于数据承载和传输
 *
 * 与数据库的关系：
 * 1. 如果使用JPA，可以添加 @Entity 注解映射到数据库表
 * 2. 如果仅用于数据传输，则作为DTO使用
 * 3. 对应 BigQuery 中的 blocks 表结构
 */
@Data  // Lombok注解：自动生成getter、setter、toString、equals、hashCode等方法
public class BlockData {

    /**
     * 区块高度（区块号）
     * 数据类型：BigInteger（大整数）
     * 原因：区块号可能非常大（以太坊当前已超过1900万）
     * 示例：19237845
     *
     * 对应的BigQuery字段：`blocks.number`
     */
    private BigInteger blockNumber;

    /**
     * 区块哈希值
     * 数据类型：String
     * 格式：0x开头的16进制字符串
     * 示例：0xd4e56740f876aef8c010b86a40d5f56745a118d0906a34e69aec8c0db1cb8fa3
     *
     * 对应的BigQuery字段：`blocks.hash`
     */
    private String blockHash;

    /**
     * 区块生成时间戳
     * 数据类型：Date（日期时间）
     * 表示区块被挖出的具体时间
     *
     * 对应的BigQuery字段：`blocks.timestamp`
     */
    private Date timestamp;

    /**
     * 矿工地址
     * 数据类型：String
     * 挖出这个区块的矿工的以太坊地址
     * 示例：0x0000000000000000000000000000000000000000
     *
     * 对应的BigQuery字段：`blocks.miner`
     */
    private String miner;

    /**
     * 区块难度
     * 数据类型：BigInteger（大整数）
     * 表示挖这个区块的计算难度
     * 以太坊使用工作量证明（PoW）时的难度值
     *
     * 对应的BigQuery字段：`blocks.difficulty`
     */
    private BigInteger difficulty;

    /**
     * 累计难度
     * 数据类型：BigInteger（大整数）
     * 从创世区块到当前区块的所有难度之和
     * 用于区块链的分叉选择规则
     *
     * 对应的BigQuery字段：`blocks.total_difficulty`
     */
    private BigInteger totalDifficulty;

    /**
     * 区块大小（字节）
     * 数据类型：BigInteger（大整数）
     * 区块数据在存储中的字节大小
     *
     * 对应的BigQuery字段：`blocks.size`
     */
    private BigInteger size;

    /**
     * 区块中使用的 Gas 总量
     * 数据类型：BigInteger（大整数）
     * Gas 是以太坊中计算和存储资源的计量单位
     * 所有交易消耗的 Gas 总和
     *
     * 对应的BigQuery字段：`blocks.gas_used`
     */
    private BigInteger gasUsed;

    /**
     * 区块 Gas 限制
     * 数据类型：BigInteger（大整数）
     * 一个区块最多可以使用的 Gas 总量
     * 用于控制区块的计算复杂度
     *
     * 对应的BigQuery字段：`blocks.gas_limit`
     */
    private BigInteger gasLimit;

    /**
     * 父区块哈希
     * 数据类型：String
     * 前一个区块的哈希值，形成区块链
     *
     * 对应的BigQuery字段：`blocks.parent_hash`
     */
    private String parentHash;

    /**
     * 额外数据
     * 数据类型：String
     * 区块中可以包含的任意数据，通常由矿工添加
     * 可能是矿工名称、消息等   
     *
     * 对应的BigQuery字段：`blocks.extra_data`
     */
    private String extraData;

    /**
     * 交易数量
     * 数据类型：Integer（整数）
     * 该区块中包含的交易数量
     *
     * 对应的BigQuery字段：`blocks.transaction_count`
     */
    private Integer transactionCount;
}