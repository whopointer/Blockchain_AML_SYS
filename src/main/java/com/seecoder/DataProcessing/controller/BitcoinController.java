// com/seecoder/DataProcessing/controller/BitcoinController.java
package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.po.ChainBlock;
import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.service.BitcoinDataService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import org.springframework.data.domain.Page;
@Slf4j
@RestController
@RequestMapping("/api/bitcoin")
@Api(tags = "比特币数据处理")
public class BitcoinController {

    @Autowired
    private BitcoinDataService bitcoinDataService;

    /**
     * 同步历史数据
     */
    @PostMapping("/sync/historical")
    public ApiResponse<String> syncHistoricalData(
            @RequestParam(required = false) Long startHeight,
            @RequestParam(required = false) Long endHeight,
            @RequestParam(required = false) Integer batchSize) {
        return bitcoinDataService.syncHistoricalData(startHeight, endHeight, batchSize);
    }

    /**
     * 查询地址余额
     */
    @GetMapping("/address/{address}/balance")
    public ApiResponse<Map<String, Object>> getAddressBalance(@PathVariable String address) {
        return bitcoinDataService.getAddressBalance(address);
    }

    /**
     * 获取区块数据（按高度范围）
     */
    @GetMapping("/blocks")
    public ApiResponse<List<ChainBlock>> getBlocks(
            @RequestParam(required = false) Long startHeight,
            @RequestParam(required = false) Long endHeight,
            @RequestParam(required = false) Integer limit) {
        return bitcoinDataService.getBlocks(startHeight, endHeight, limit);
    }

    /**
     * 按时间范围获取区块
     */
    @GetMapping("/blocks/by-time")
    public ApiResponse<List<ChainBlock>> getBlocksByTime(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(required = false) Integer limit) {
        return bitcoinDataService.getBlocksByTime(startTime, endTime, limit);
    }

    /**
     * 获取交易数据（按区块高度）
     */
    @GetMapping("/transactions")
    public ApiResponse<List<ChainTx>> getTransactions(
            @RequestParam Long blockHeight,
            @RequestParam(required = false) Integer limit,
            @RequestParam(required = false) Integer offset) {
        return bitcoinDataService.getTransactions(blockHeight, limit, offset);
    }

    /**
     * 按时间范围获取交易
     */
    @GetMapping("/transactions/by-time")
    public ApiResponse<List<ChainTx>> getTransactionsByTime(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(required = false) Integer limit) {
        return bitcoinDataService.getTransactionsByTime(startTime, endTime, limit);
    }

    /**
     * 获取交易详情
     */
    @GetMapping("/transaction/{txHash}")
    public ApiResponse<Map<String, Object>> getTransactionDetail(@PathVariable String txHash) {
        return bitcoinDataService.getTransactionDetail(txHash);
    }

    /**
     * 获取地址信息（包括余额等）
     */
    @GetMapping("/address/{address}")
    public ApiResponse<Map<String, Object>> getAddressInfo(@PathVariable String address) {
        return bitcoinDataService.getAddressInfo(address);
    }

    /**
     * 导出区块数据到CSV（按高度范围）
     */
    @GetMapping("/export/blocks")
    public ApiResponse<String> exportBlocksToCsv(
            @RequestParam(required = false) Long startHeight,
            @RequestParam(required = false) Long endHeight) {
        return bitcoinDataService.exportBlocksToCsv(startHeight, endHeight);
    }

    /**
     * 导出交易数据到CSV（按时间范围）
     */
    @GetMapping("/export/transactions")
    public ApiResponse<String> exportTransactionsToCsv(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {
        return bitcoinDataService.exportTransactionsToCsv(startTime, endTime);
    }

    /**
     * 获取区块链统计信息
     */
    @GetMapping("/stats")
    public ApiResponse<Map<String, Object>> getBlockchainStats() {
        return bitcoinDataService.getBlockchainStats();
    }

    /**
     * 获取最新数据状态（数据库与BigQuery对比）
     */
    @GetMapping("/latest")
    public ApiResponse<Map<String, Object>> getLatestData() {
        return bitcoinDataService.getLatestData();
    }

    /**
     * 同步最新数据
     */
    @PostMapping("/sync/latest")
    public ApiResponse<String> syncLatestData(@RequestParam(required = false) Integer blocksToSync) {
        return bitcoinDataService.syncLatestData(blocksToSync);
    }

    /**
     * 分页查询区块
     */
    @GetMapping("/blocks/page")
    public ApiResponse<Page<ChainBlock>> getBlocksPage(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size) {
        return bitcoinDataService.getBlocksPage(page, size);
    }

    /**
     * 分页查询交易
     */
    @GetMapping("/transactions/page")
    public ApiResponse<Page<ChainTx>> getTransactionsPage(
            @RequestParam(required = false) Integer page,
            @RequestParam(required = false) Integer size) {
        return bitcoinDataService.getTransactionsPage(page, size);
    }

    /**
     * 获取指定地址的所有交易
     */
    @GetMapping("/address/{address}/transactions")
    public ApiResponse<List<ChainTx>> getTransactionsByAddress(
            @PathVariable String address,
            @RequestParam(required = false) Integer limit) {
        return bitcoinDataService.getTransactionsByAddress(address, limit);
    }

    /**
     * 获取每日统计（最近N天）
     */
    @GetMapping("/stats/daily")
    public ApiResponse<List<Map<String, Object>>> getDailyStats(@RequestParam(required = false) Integer days) {
        return bitcoinDataService.getDailyStats(days);
    }

    /**
     * 清除所有缓存
     */
    @PostMapping("/cache/clear")
    public ApiResponse<Void> clearAllCache() {
        bitcoinDataService.clearAllCache();
        return ApiResponse.success(null, null); // 假设ApiResponse有静态success方法，可能需要调整
    }

    /**
     * 获取当前数据库中最新区块高度
     */
    @GetMapping("/blocknumber")
    public ApiResponse<Long> getBlockNumber() {
        return bitcoinDataService.getBlockNumber();
    }
}