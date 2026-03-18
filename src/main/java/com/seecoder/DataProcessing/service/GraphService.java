// com/seecoder/DataProcessing/service/GraphService.java
package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.ChainTxInput;
import com.seecoder.DataProcessing.po.ChainTxOutput;
import com.seecoder.DataProcessing.vo.ApiResponse;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Set;

public interface GraphService {
    public ApiResponse<Map<String, Object>> testNeo4jConnection();
    // 创建图谱快照，将快照对象持久化到 MySQL
    ApiResponse<com.seecoder.DataProcessing.po.GraphSnapshot> createGraphSnapshot(com.seecoder.DataProcessing.po.GraphSnapshot snapshot);
    // 保存交易到图数据库（两种关系都保存）
    void saveTransactionToGraph(ChainTx chainTx);

    // 批量保存交易
    void saveTransactionsToGraph(List<ChainTx> chainTxs);

    // 查找交易路径
    ApiResponse<Map<String, Object>> findNhopTransactionPath(String fromAddress, String toAddress);

    // 查找地址N跳内的所有关联地址
    ApiResponse<Map<String, Object>> findAddressesWithinNHops(String address, Integer maxHops);

    // 获取地址的转账统计
    ApiResponse<Map<String, Object>> getAddressTransferStats(String address);

    // 获取两个地址之间的转账统计
    ApiResponse<Map<String, Object>> getTransferStatsBetweenAddresses(String fromAddress, String toAddress);

    // 查找大额转账
    ApiResponse<List<Map<String, Object>>> findLargeTransfers(BigDecimal minAmount, LocalDateTime startTime, LocalDateTime endTime, Integer limit);

    // 分析地址的交易模式
    ApiResponse<Map<String, Object>> analyzeAddressPattern(String address, Integer depth);

    // 获取地址的直接关联地址
    ApiResponse<Map<String, Object>> getDirectConnections(String address);

    // 更新地址的风险等级
    ApiResponse<Void> updateAddressRiskLevel(String address, Integer riskLevel);

    // 为地址打标签
    ApiResponse<Void> tagAddress(String address, String tag);

    /**
     * 获取指定地址在时间范围内的直接交易对手地址
     */
    Set<String> getNeighborAddresses(String address, LocalDateTime startTime, LocalDateTime endTime);

    /**
     * 获取指定地址在时间范围内的交易哈希
     */
    Set<String> getTransactionHashes(String address, LocalDateTime startTime, LocalDateTime endTime);

    /**
     * 保存一笔比特币交易到图数据库（UTXO模型）
     * @param tx      交易主体
     * @param inputs  交易输入列表
     * @param outputs 交易输出列表
     */
    void saveBitcoinTransactionToGraph(ChainTx tx, List<ChainTxInput> inputs, List<ChainTxOutput> outputs);

    /**
     * 批量保存多笔比特币交易到图数据库
     * @param txs        交易列表
     * @param inputsMap  交易哈希 -> 输入列表 映射
     * @param outputsMap 交易哈希 -> 输出列表 映射
     */
    void saveBitcoinTransactionsToGraph(List<ChainTx> txs,
                                        Map<String, List<ChainTxInput>> inputsMap,
                                        Map<String, List<ChainTxOutput>> outputsMap);
    // 清理图数据
    void cleanGraphData(String chain);

    // 获取所有图谱快照
    ApiResponse<List<com.seecoder.DataProcessing.po.GraphSnapshot>> getAllGraphSnapshots();

    // 修改图谱快照信息
    ApiResponse<com.seecoder.DataProcessing.po.GraphSnapshot> updateGraphSnapshot(Long id, com.seecoder.DataProcessing.po.GraphSnapshot snapshot);

    // 删除图谱快照
    ApiResponse<Void> deleteGraphSnapshot(Long id);
}