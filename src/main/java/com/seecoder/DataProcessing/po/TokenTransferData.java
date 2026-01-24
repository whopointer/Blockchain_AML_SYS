package com.seecoder.DataProcessing.po;

import lombok.Data;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Date;

/**
 * Token 转账数据实体类
 * 作用：表示 ERC20/ERC721/ERC1155 等 Token 的转账记录
 * 数据来源：以太坊等区块链的 Transfer 事件日志
 *
 * 重要概念：
 * 1. Token：基于区块链的可编程资产
 * 2. Transfer：Token从一个地址转移到另一个地址
 * 3. Event Log：智能合约发出的事件，记录在区块中
 */
@Data
public class TokenTransferData {

    /**
     * 交易哈希
     * 数据类型：String
     * 包含此 Token 转账的交易的哈希值
     * 注意：一笔交易可能包含多个 Token 转账（批量转账）
     *
     * 对应的BigQuery字段：`token_transfers.transaction_hash`
     */
    private String txHash;

    /**
     * 日志索引
     * 数据类型：String
     * 在交易的事件日志中的索引位置
     * 用于区分同一交易中的多个事件
     *
     * 对应的BigQuery字段：`token_transfers.log_index`
     */
    private String logIndex;

    /**
     * Token 合约地址
     * 数据类型：String
     * 发行该 Token 的智能合约地址
     * 不同 Token 有不同的合约地址
     *
     * 对应的BigQuery字段：`token_transfers.token_address`
     */
    private String contractAddress;

    /**
     * 发送方地址
     * 数据类型：String
     * Token 的转出地址
     *
     * 对应的BigQuery字段：`token_transfers.from_address`
     */
    private String fromAddress;

    /**
     * 接收方地址
     * 数据类型：String
     * Token 的转入地址
     *
     * 对应的BigQuery字段：`token_transfers.to_address`
     */
    private String toAddress;

    /**
     * 转账数量/金额
     * 数据类型：BigDecimal（高精度小数）
     * 原因：Token 可能有不同的小数位数（如 USDT 有6位小数）
     * 注意：对于 NFT（ERC721），value 可能为 1 或 tokenId
     *
     * 对应的BigQuery字段：`token_transfers.value`
     */
    private BigDecimal value;

    /**
     * Token 符号
     * 数据类型：String
     * Token 的缩写符号，如 USDT、DAI、ETH
     * 注意：BigQuery 公共数据集中可能不包含此字段，需要从其他来源获取
     */
    private String tokenSymbol;

    /**
     * Token 小数位数
     * 数据类型：BigInteger（大整数）
     * Token 的最小单位，决定 value 字段的精度
     * 示例：USDT 的 decimals 为 6，1 USDT = 1000000 个最小单位
     */
    private BigInteger decimals;

    /**
     * Token 标准
     * 数据类型：String
     * Token 遵循的 ERC 标准：ERC20、ERC721、ERC1155 等
     * 不同的标准有不同的功能和特性
     */
    private String tokenStandard;

    /**
     * 区块时间戳
     * 数据类型：Date
     * 包含该转账的区块的生成时间
     *
     * 对应的BigQuery字段：`token_transfers.block_timestamp`
     */
    private Date blockTimestamp;

    /**
     * 区块高度
     * 数据类型：BigInteger（大整数）
     * 包含该转账的区块的编号
     *
     * 对应的BigQuery字段：`token_transfers.block_number`
     */
    private BigInteger blockNumber;

    /**
     * 区块链网络
     * 数据类型：String
     * 数据来源的区块链网络
     * 示例：ETHEREUM_MAINNET、BSC_MAINNET
     * 用于区分不同链的数据
     */
    private String network;
}