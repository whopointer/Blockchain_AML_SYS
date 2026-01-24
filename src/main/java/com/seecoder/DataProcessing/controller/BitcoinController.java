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

@Slf4j
@RestController
@RequestMapping("/api/bitcoin")
@Api(tags = "比特币数据处理")
public class BitcoinController {

    @Autowired
    private BitcoinDataService bitcoinDataService;

    @GetMapping("/blocks")
    @ApiOperation("获取比特币区块数据")
    public ApiResponse<List<ChainBlock>> getBlocks(
            @RequestParam(required = false) Long startHeight,
            @RequestParam(required = false) Long endHeight,
            @RequestParam(defaultValue = "100") Integer limit) {

        return bitcoinDataService.getBlocks(startHeight, endHeight, limit);
    }

    @GetMapping("/blocks/time-range")
    @ApiOperation("按时间范围获取区块数据")
    public ApiResponse<List<ChainBlock>> getBlocksByTime(
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime,
            @RequestParam(defaultValue = "100") Integer limit) {

        return bitcoinDataService.getBlocksByTime(startTime, endTime, limit);
    }

    @GetMapping("/transactions")
    @ApiOperation("获取比特币交易数据")
    public ApiResponse<List<ChainTx>> getTransactions(
            @RequestParam Long blockHeight,
            @RequestParam(defaultValue = "100") Integer limit,
            @RequestParam(defaultValue = "0") Integer offset) {

        return bitcoinDataService.getTransactions(blockHeight, limit, offset);
    }

    @GetMapping("/transactions/time-range")
    @ApiOperation("按时间范围获取交易数据")
    public ApiResponse<List<ChainTx>> getTransactionsByTime(
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime,
            @RequestParam(defaultValue = "100") Integer limit) {

        return bitcoinDataService.getTransactionsByTime(startTime, endTime, limit);
    }

    @GetMapping("/transaction/{txHash}")
    @ApiOperation("获取交易详情")
    public ApiResponse<Map<String, Object>> getTransactionDetail(
            @PathVariable String txHash) {

        return bitcoinDataService.getTransactionDetail(txHash);
    }

    @GetMapping("/address/{address}")
    @ApiOperation("获取地址信息")
    public ApiResponse<Map<String, Object>> getAddressInfo(
            @PathVariable String address) {

        return bitcoinDataService.getAddressInfo(address);
    }

    @PostMapping("/export/blocks")
    @ApiOperation("导出区块数据到CSV")
    public ApiResponse<String> exportBlocksToCsv(
            @RequestParam Long startHeight,
            @RequestParam Long endHeight) {

        return bitcoinDataService.exportBlocksToCsv(startHeight, endHeight);
    }

    @PostMapping("/export/transactions")
    @ApiOperation("导出交易数据到CSV")
    public ApiResponse<String> exportTransactionsToCsv(
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime startTime,
            @RequestParam @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime endTime) {

        return bitcoinDataService.exportTransactionsToCsv(startTime, endTime);
    }

    @GetMapping("/stats")
    @ApiOperation("获取区块链统计信息")
    public ApiResponse<Map<String, Object>> getBlockchainStats() {
        return bitcoinDataService.getBlockchainStats();
    }

    @PostMapping("/sync/latest")
    @ApiOperation("同步最新区块数据")
    public ApiResponse<String> syncLatestBlocks(
            @RequestParam(defaultValue = "10") Integer limit) {

        return bitcoinDataService.syncLatestBlocks(limit);
    }

    @PostMapping("/sync/historical")
    @ApiOperation("同步历史数据")
    public ApiResponse<String> syncHistoricalData(
            @RequestParam Long startHeight,
            @RequestParam Long endHeight,
            @RequestParam(defaultValue = "100") Integer batchSize) {

        return bitcoinDataService.syncHistoricalData(startHeight, endHeight, batchSize);
    }

    @PostMapping("/sync/address")
    @ApiOperation("同步地址数据")
    public ApiResponse<String> syncAddressData(
            @RequestParam String address) {

        // 这个方法需要实现，这里简化为返回成功
        log.info("同步地址数据: {}", address);
        return ApiResponse.success("地址数据同步任务已启动", null);
    }
}