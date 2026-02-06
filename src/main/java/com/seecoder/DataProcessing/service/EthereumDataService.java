// com/seecoder/DataProcessing/service/EthereumDataService.java
package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.po.ChainBlock;
import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.vo.ApiResponse;
import org.springframework.data.domain.Page;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public interface EthereumDataService {
    public ApiResponse<String> testGraphConnection();

    // ============= 区块相关方法 =============

    /**
     * 获取区块列表（按高度范围）
     */
    ApiResponse<List<ChainBlock>> getBlocks(Long startHeight, Long endHeight, Integer limit);

    /**
     * 获取区块列表（按时间范围）
     */
    ApiResponse<List<ChainBlock>> getBlocksByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit);

    /**
     * 获取区块分页数据
     */
    ApiResponse<Page<ChainBlock>> getBlocksPage(Integer page, Integer size);

    // ============= 交易相关方法 =============

    /**
     * 获取交易列表
     */
    ApiResponse<List<ChainTx>> getTransactions(Long blockHeight, Integer limit, Integer offset);

    /**
     * 获取交易列表（按时间范围）
     */
    ApiResponse<List<ChainTx>> getTransactionsByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit);

    /**
     * 获取交易分页数据
     */
    ApiResponse<Page<ChainTx>> getTransactionsPage(Integer page, Integer size);

    /**
     * 获取交易详情
     */
    ApiResponse<Map<String, Object>> getTransactionDetail(String txHash);

    /**
     * 获取地址相关交易
     */
    ApiResponse<List<ChainTx>> getTransactionsByAddress(String address, Integer limit);

    // ============= 地址相关方法 =============

    /**
     * 获取地址信息
     */
    ApiResponse<Map<String, Object>> getAddressInfo(String address);

    /**
     * 获取地址余额
     */
    ApiResponse<Map<String, Object>> getAddressBalance(String address);

    // ============= 导出方法 =============

    /**
     * 导出区块数据到CSV
     */
    ApiResponse<String> exportBlocksToCsv(Long startHeight, Long endHeight);

    /**
     * 导出交易数据到CSV
     */
    ApiResponse<String> exportTransactionsToCsv(LocalDateTime startTime, LocalDateTime endTime);

    // ============= 统计方法 =============

    /**
     * 获取区块链统计信息
     */
    ApiResponse<Map<String, Object>> getBlockchainStats();

    /**
     * 获取每日统计
     */
    ApiResponse<List<Map<String, Object>>> getDailyStats(Integer days);

    /**
     * 获取最新区块高度
     */
    ApiResponse<Long> getBlockNumber();

    // ============= 同步方法 =============

    /**
     * 同步最新数据
     */
    ApiResponse<String> syncLatestData(Integer blocksToSync);

    /**
     * 同步历史数据（按日期）
     */
    ApiResponse<String> syncHistoricalData(LocalDate startDate, LocalDate endDate, Integer batchDays);

    /**
     * 获取最新数据状态
     */
    ApiResponse<Map<String, Object>> getLatestData();

    // ============= 缓存清理方法 =============

    /**
     * 清理所有缓存
     */
    void clearAllCache();
}