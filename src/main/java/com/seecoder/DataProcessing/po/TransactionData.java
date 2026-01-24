package com.seecoder.DataProcessing.po;

import lombok.Data;
import java.math.BigInteger;
import java.util.Date;

/**
 * 区块链交易数据实体类
 * 作用：表示区块链上的一笔交易记录
 * 交易类型：普通转账、合约调用、合约部署等
 *
 * 交易生命周期：
 * 1. 用户签名交易并广播
 * 2. 矿工打包交易到区块
 * 3. 区块上链，交易确认
 */
@Data
public class TransactionData {

    /**
     * 交易哈希
     * 数据类型：String
     * 交易的唯一标识符，由交易内容通过哈希算法生成
     * 格式：0x开头的64位16进制字符串
     * 示例：0x88df016429689c079f3b2f6ad39fa052532c56795b733da78a91ebe6a713944b
     *
     * 对应的BigQuery字段：`transactions.transaction_hash`
     */
    private String txHash;

    /**
     * 区块哈希
     * 数据类型：String
     * 包含该交易的区块的哈希值
     *
     * 对应的BigQuery字段：`transactions.block_hash`
     */
    private String blockHash;

    /**
     * 区块高度
     * 数据类型：BigInteger（大整数）
     * 包含该交易的区块的编号
     *
     * 对应的BigQuery字段：`transactions.block_number`
     */
    private BigInteger blockNumber;

    /**
     * 区块时间戳
     * 数据类型：Date
     * 包含该交易的区块的生成时间
     *
     * 对应的BigQuery字段：`transactions.block_timestamp`
     */
    private Date blockTimestamp;

    /**
     * 发送方地址
     * 数据类型：String
     * 交易的发起者，需要支付 Gas 费
     * 由私钥签名证明所有权
     *
     * 对应的BigQuery字段：`transactions.from_address`
     */
    private String fromAddress;

    /**
     * 接收方地址
     * 数据类型：String
     * 交易的接收者，可以是普通地址或合约地址
     * 对于合约部署交易，此字段为空
     *
     * 对应的BigQuery字段：`transactions.to_address`
     */
    private String toAddress;

    /**
     * 合约地址（如果交易是合约创建）
     * 数据类型：String
     * 仅当交易是合约创建（constructor）时有效
     * 新部署的合约的地址
     *
     * 对应的BigQuery字段：`transactions.contract_address`
     */
    private String contractAddress;

    /**
     * 交易金额（以太币数量）
     * 数据类型：BigInteger（大整数）
     * 从 from 地址转移到 to 地址的以太币数量
     * 单位：wei（1 ETH = 10^18 wei）
     *
     * 对应的BigQuery字段：`transactions.value`
     */
    private BigInteger value;

    /**
     * Gas 单价
     * 数据类型：BigInteger（大整数）
     * 每单位 Gas 的价格，单位：wei
     * 影响交易被打包的速度：价格越高，优先级越高
     *
     * 对应的BigQuery字段：`transactions.gas_price`
     */
    private BigInteger gasPrice;

    /**
     * 实际使用的 Gas 数量
     * 数据类型：BigInteger（大整数）
     * 交易实际消耗的 Gas 数量
     * 实际费用 = gasPrice × gasUsed
     *
     * 对应的BigQuery字段：`transactions.gas_used`
     */
    private BigInteger gasUsed;

    /**
     * 输入数据
     * 数据类型：String
     * 交易调用的函数和参数
     * 对于普通转账为空，对于合约调用包含函数签名和参数
     * 格式：0x开头的16进制字符串
     *
     * 对应的BigQuery字段：`transactions.input`
     */
    private String inputData;

    /**
     * 交易类型
     * 数据类型：String
     * 以太坊交易类型：0（传统交易）、1（访问列表）、2（EIP-1559动态费用）
     * EIP-1559 引入了新的交易类型，优化 Gas 费用机制
     *
     * 对应的BigQuery字段：`transactions.transaction_type`
     */
    private String txType;

    /**
     * 交易状态
     * 数据类型：Integer（整数）
     * 0：失败（reverted）
     * 1：成功（success）
     * 交易可能因为各种原因失败：Gas不足、合约执行异常等
     *
     * 对应的BigQuery字段：`transactions.status`
     */
    private Integer status;

    /**
     * 区块链网络
     * 数据类型：String
     * 数据来源的区块链网络
     * 用于区分不同链的数据
     */
    private String network;
}