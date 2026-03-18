// com/seecoder/DataProcessing/service/BitcoinDataService.java
package com.seecoder.DataProcessing.service;

import com.seecoder.DataProcessing.po.ChainBlock;
import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.ChainTxInput;
import com.seecoder.DataProcessing.po.ChainTxOutput;
import com.seecoder.DataProcessing.vo.ApiResponse;
import org.springframework.data.domain.Page;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public interface BitcoinDataService {
    ApiResponse<String> syncHistoricalData(Long startHeight, Long endHeight, Integer batchSize);

    // 查询地址余额
    ApiResponse<Map<String, Object>> getAddressBalance(String address);


    // 获取区块数据
    ApiResponse<List<ChainBlock>> getBlocks(Long startHeight, Long endHeight, Integer limit);

    ApiResponse<List<ChainBlock>> getBlocksByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit);

    // 获取交易数据
    ApiResponse<List<ChainTx>> getTransactions(Long blockHeight, Integer limit, Integer offset);

    ApiResponse<List<ChainTx>> getTransactionsByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit);

    // 获取交易详情
    ApiResponse<Map<String, Object>> getTransactionDetail(String txHash);

    // 获取地址信息
    ApiResponse<Map<String, Object>> getAddressInfo(String address);

    // 数据导出
    ApiResponse<String> exportBlocksToCsv(Long startHeight, Long endHeight);

    ApiResponse<String> exportTransactionsToCsv(LocalDateTime startTime, LocalDateTime endTime);

    // 数据统计
    ApiResponse<Map<String, Object>> getBlockchainStats();


    ApiResponse<Map<String, Object>> getLatestData();

    ApiResponse<String> syncLatestData(Integer blocksToSync);

    ApiResponse<Page<ChainBlock>> getBlocksPage(Integer page, Integer size);

    ApiResponse<Page<ChainTx>> getTransactionsPage(Integer page, Integer size);

    ApiResponse<List<ChainTx>> getTransactionsByAddress(String address, Integer limit);


    ApiResponse<List<Map<String, Object>>> getDailyStats(Integer days);

    void clearAllCache();

    ApiResponse<Long> getBlockNumber();
}
