// com/seecoder/DataProcessing/controller/EthereumDataController.java
package com.seecoder.DataProcessing.controller;

import com.seecoder.DataProcessing.po.ChainBlock;
import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.service.EthereumDataService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/ethereum")
@CrossOrigin(origins = "*")
public class EthereumController {

    @Autowired
    private EthereumDataService ethereumDataService;

    // ============= 区块相关接口 =============

    /**
     * 获取区块列表（按高度范围）
     * GET /api/ethereum/blocks?startHeight=100000&endHeight=100100&limit=10
     */
    @GetMapping("/blocks")
    public ResponseEntity<ApiResponse<List<ChainBlock>>> getBlocks(
            @RequestParam(required = false) Long startHeight,
            @RequestParam(required = false) Long endHeight,
            @RequestParam(defaultValue = "100") Integer limit) {

        ApiResponse<List<ChainBlock>> response = ethereumDataService.getBlocks(startHeight, endHeight, limit);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取区块列表（按时间范围）
     * GET /api/ethereum/blocks/by-time?startTime=2024-01-01T00:00:00&endTime=2024-01-02T00:00:00&limit=10
     */
    @GetMapping("/blocks/by-time")
    public ResponseEntity<ApiResponse<List<ChainBlock>>> getBlocksByTime(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(defaultValue = "100") Integer limit) {

        ApiResponse<List<ChainBlock>> response = ethereumDataService.getBlocksByTime(startTime, endTime, limit);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取区块分页数据
     * GET /api/ethereum/blocks/page?page=0&size=20
     */
    @GetMapping("/blocks/page")
    public ResponseEntity<ApiResponse<Page<ChainBlock>>> getBlocksPage(
            @RequestParam(defaultValue = "0") Integer page,
            @RequestParam(defaultValue = "20") Integer size) {

        ApiResponse<Page<ChainBlock>> response = ethereumDataService.getBlocksPage(page, size);
        return ResponseEntity.ok(response);
    }

    // ============= 交易相关接口 =============

    /**
     * 获取交易列表
     * GET /api/ethereum/transactions?blockHeight=100000&limit=10&offset=0
     */
    @GetMapping("/transactions")
    public ResponseEntity<ApiResponse<List<ChainTx>>> getTransactions(
            @RequestParam Long blockHeight,
            @RequestParam(defaultValue = "100") Integer limit,
            @RequestParam(defaultValue = "0") Integer offset) {

        ApiResponse<List<ChainTx>> response = ethereumDataService.getTransactions(blockHeight, limit, offset);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取交易列表（按时间范围）
     * GET /api/ethereum/transactions/by-time?startTime=2024-01-01T00:00:00&endTime=2024-01-02T00:00:00&limit=10
     */
    @GetMapping("/transactions/by-time")
    public ResponseEntity<ApiResponse<List<ChainTx>>> getTransactionsByTime(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime,
            @RequestParam(defaultValue = "100") Integer limit) {

        ApiResponse<List<ChainTx>> response = ethereumDataService.getTransactionsByTime(startTime, endTime, limit);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取交易分页数据
     * GET /api/ethereum/transactions/page?page=0&size=20
     */
    @GetMapping("/transactions/page")
    public ResponseEntity<ApiResponse<Page<ChainTx>>> getTransactionsPage(
            @RequestParam(defaultValue = "0") Integer page,
            @RequestParam(defaultValue = "20") Integer size) {

        ApiResponse<Page<ChainTx>> response = ethereumDataService.getTransactionsPage(page, size);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取地址相关交易
     * GET /api/ethereum/address/{address}/transactions?limit=20
     */
    @GetMapping("/address/{address}/transactions")
    public ResponseEntity<ApiResponse<List<ChainTx>>> getTransactionsByAddress(
            @PathVariable String address,
            @RequestParam(defaultValue = "20") Integer limit) {

        ApiResponse<List<ChainTx>> response = ethereumDataService.getTransactionsByAddress(address, limit);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取交易详情
     * GET /api/ethereum/transaction/{txHash}
     */
    @GetMapping("/transaction/{txHash}")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getTransactionDetail(
            @PathVariable String txHash) {

        ApiResponse<Map<String, Object>> response = ethereumDataService.getTransactionDetail(txHash);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取地址信息
     * GET /api/ethereum/address/{address}
     */
    @GetMapping("/address/{address}")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getAddressInfo(
            @PathVariable String address) {

        ApiResponse<Map<String, Object>> response = ethereumDataService.getAddressInfo(address);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取地址余额
     * GET /api/ethereum/address/{address}/balance
     */
    @GetMapping("/address/{address}/balance")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getAddressBalance(
            @PathVariable String address) {

        ApiResponse<Map<String, Object>> response = ethereumDataService.getAddressBalance(address);
        return ResponseEntity.ok(response);
    }

    // ============= 统计接口 =============

    /**
     * 获取区块链统计信息
     * GET /api/ethereum/stats
     */
    @GetMapping("/stats")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getBlockchainStats() {

        ApiResponse<Map<String, Object>> response = ethereumDataService.getBlockchainStats();
        return ResponseEntity.ok(response);
    }

    /**
     * 获取每日统计
     * GET /api/ethereum/stats/daily?days=7
     */
    @GetMapping("/stats/daily")
    public ResponseEntity<ApiResponse<List<Map<String, Object>>>> getDailyStats(
            @RequestParam(defaultValue = "7") Integer days) {

        ApiResponse<List<Map<String, Object>>> response = ethereumDataService.getDailyStats(days);
        return ResponseEntity.ok(response);
    }

    /**
     * 获取最新区块高度
     * GET /api/ethereum/block-number
     */
    @GetMapping("/block-number")
    public ResponseEntity<ApiResponse<Long>> getBlockNumber() {

        ApiResponse<Long> response = ethereumDataService.getBlockNumber();
        return ResponseEntity.ok(response);
    }

    /**
     * 获取最新数据状态
     * GET /api/ethereum/latest-status
     */
    @GetMapping("/latest-status")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getLatestData() {

        ApiResponse<Map<String, Object>> response = ethereumDataService.getLatestData();
        return ResponseEntity.ok(response);
    }

    // ============= 同步接口 =============

    /**
     * 同步最新数据
     * POST /api/ethereum/sync/latest?blocksToSync=100
     */
    @PostMapping("/sync/latest")
    public ResponseEntity<ApiResponse<String>> syncLatestData(
            @RequestParam(defaultValue = "100") Integer blocksToSync) {

        ApiResponse<String> response = ethereumDataService.syncLatestData(blocksToSync);
        return ResponseEntity.ok(response);
    }

    /**
     * 同步历史数据（按日期）
     * POST /api/ethereum/sync/historical?startDate=2023-01-01&endDate=2023-01-07&batchDays=1
     */
    @PostMapping("/sync/historical")
    public ResponseEntity<ApiResponse<String>> syncHistoricalData(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate,
            @RequestParam(defaultValue = "1") Integer batchDays) {

        ApiResponse<String> response = ethereumDataService.syncHistoricalData(startDate, endDate, batchDays);
        return ResponseEntity.ok(response);
    }

    /**
     * 清理缓存
     * POST /api/ethereum/cache/clear
     */
    @PostMapping("/cache/clear")
    public ResponseEntity<ApiResponse<String>> clearAllCache() {
        ethereumDataService.clearAllCache();
        return ResponseEntity.ok(ApiResponse.success("缓存已清理", null));
    }

    // ============= 导出接口 =============

    /**
     * 导出区块数据到CSV
     * GET /api/ethereum/export/blocks?startHeight=100000&endHeight=100100
     */
    @GetMapping("/export/blocks")
    public ResponseEntity<ApiResponse<String>> exportBlocksToCsv(
            @RequestParam Long startHeight,
            @RequestParam Long endHeight) {

        ApiResponse<String> response = ethereumDataService.exportBlocksToCsv(startHeight, endHeight);
        return ResponseEntity.ok(response);
    }

    /**
     * 导出交易数据到CSV
     * GET /api/ethereum/export/transactions?startTime=2024-01-01T00:00:00&endTime=2024-01-02T00:00:00
     */
    @GetMapping("/export/transactions")
    public ResponseEntity<ApiResponse<String>> exportTransactionsToCsv(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startTime,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endTime) {

        ApiResponse<String> response = ethereumDataService.exportTransactionsToCsv(startTime, endTime);
        return ResponseEntity.ok(response);
    }
}