// com/seecoder/DataProcessing/service/GraphService.java
package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.vo.ApiResponse;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public interface GraphService {

    // 保存交易到图数据库（两种关系都保存）
    void saveTransactionToGraph(ChainTx chainTx);

    // 批量保存交易
    void saveTransactionsToGraph(List<ChainTx> chainTxs);

    // 查找N跳交易路径
    ApiResponse<List<Map<String, Object>>> findNhopTransactionPath(String fromAddress, String toAddress, Integer maxHops);

    // 查找地址N跳内的所有关联地址
    ApiResponse<List<Map<String, Object>>> findAddressesWithinNHops(String address, Integer maxHops);

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

    // 清理图数据
    void cleanGraphData(String chain);
}