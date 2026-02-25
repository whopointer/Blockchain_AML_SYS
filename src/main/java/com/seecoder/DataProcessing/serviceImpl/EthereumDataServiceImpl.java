// com/seecoder/DataProcessing/serviceImpl/EthereumDataServiceImpl.java
package com.seecoder.DataProcessing.serviceImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.cloud.bigquery.*;
import com.seecoder.DataProcessing.po.*;
import com.seecoder.DataProcessing.repository.*;
import com.seecoder.DataProcessing.service.EthereumDataService;
import com.seecoder.DataProcessing.service.GraphService;
import com.seecoder.DataProcessing.service.MinIOService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.math.BigDecimal;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;

@Slf4j
@Service
public class EthereumDataServiceImpl implements EthereumDataService {

    @Autowired
    private BigQuery bigQuery;

    @Autowired
    private ChainBlockRepository chainBlockRepository;

    @Autowired
    private ChainTxRepository chainTxRepository;

    @Autowired
    private ChainTxInputRepository chainTxInputRepository;

    @Autowired
    private ChainTxOutputRepository chainTxOutputRepository;

    @Autowired
    private GraphService graphService;

    @Autowired
    private MinIOService minIOService;

    @Autowired
    private ObjectMapper objectMapper;

    @Value("${minio.archive.enabled:true}")
    private boolean minioArchiveEnabled;

    private static final DateTimeFormatter BIGQUERY_TIMESTAMP_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final DateTimeFormatter DATE_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd");
    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyyMMdd_HHmmss");

    // 常量定义
    private static final int MAX_QUERY_DAYS = 7;
    private static final int MAX_BLOCKS_PER_QUERY = 1000;
    private static final int MAX_TRANSACTIONS_PER_QUERY = 10000;
    private static final String CHAIN_ETH = "ETH";

    // ============= MinIO归档相关方法 =============

    /**
     * 归档BigQuery查询结果到MinIO
     */
    private void archiveBigQueryResult(String queryType, TableResult result) {
        if (!minioArchiveEnabled) {
            return;
        }

        CompletableFuture.runAsync(() -> {
            try {
                List<Map<String, Object>> dataList = new ArrayList<>();
                for (FieldValueList row : result.iterateAll()) {
                    Map<String, Object> rowData = new HashMap<>();
                    for (Field field : result.getSchema().getFields()) {
                        String fieldName = field.getName();
                        FieldValue fieldValue = row.get(fieldName);

                        if (!fieldValue.isNull()) {
                            switch (fieldValue.getAttribute()) {
                                case PRIMITIVE:
                                    rowData.put(fieldName, fieldValue.getValue());
                                    break;
                                case RECORD:
                                    // 处理嵌套记录
                                    break;
                                default:
                                    rowData.put(fieldName, fieldValue.getValue());
                            }
                        }
                    }
                    dataList.add(rowData);
                }

                String jsonResponse = objectMapper.writeValueAsString(dataList);
                minIOService.archiveRawBigQueryResponse(queryType, jsonResponse);

            } catch (Exception e) {
                log.error("归档BigQuery结果失败: {}", queryType, e);
            }
        });
    }

    /**
     * 归档单笔交易数据到MinIO
     */
    private void archiveTransactionData(ChainTx tx) {
        if (!minioArchiveEnabled) {
            return;
        }

        CompletableFuture.runAsync(() -> {
            try {
                Map<String, Object> txData = new HashMap<>();
                txData.put("hash", tx.getTxHash());
                txData.put("block_number", tx.getBlockHeight());
                txData.put("from_address", tx.getFromAddress());
                txData.put("to_address", tx.getToAddress());
                txData.put("value", tx.getTotalOutput());
                txData.put("fee", tx.getFee());
                txData.put("block_time", tx.getBlockTime());
                txData.put("tx_index", tx.getTxIndex());
                txData.put("status", tx.getStatus());

                String jsonData = objectMapper.writeValueAsString(txData);
                minIOService.archiveTransactionData(tx.getTxHash(), jsonData);

            } catch (Exception e) {
                log.error("归档交易数据失败: {}", tx.getTxHash(), e);
            }
        });
    }

    /**
     * 归档区块数据到MinIO
     */
    private void archiveBlockData(ChainBlock block) {
        if (!minioArchiveEnabled) {
            return;
        }

        CompletableFuture.runAsync(() -> {
            try {
                Map<String, Object> blockData = new HashMap<>();
                blockData.put("height", block.getHeight());
                blockData.put("block_hash", block.getBlockHash());
                blockData.put("prev_block_hash", block.getPrevBlockHash());
                blockData.put("block_time", block.getBlockTime());
                blockData.put("tx_count", block.getTxCount());
                blockData.put("size_bytes", block.getRawSizeBytes());

                String jsonData = objectMapper.writeValueAsString(blockData);
                minIOService.archiveBlockData(block.getHeight(), jsonData);

            } catch (Exception e) {
                log.error("归档区块数据失败: {}", block.getHeight(), e);
            }
        });
    }

    /**
     * 添加日志归档方法
     */
    private void archiveSyncLog(String operation, String details, boolean success) {
        if (!minioArchiveEnabled) {
            return;
        }

        CompletableFuture.runAsync(() -> {
            try {
                Map<String, Object> logEntry = new HashMap<>();
                logEntry.put("timestamp", LocalDateTime.now().toString());
                logEntry.put("operation", operation);
                logEntry.put("details", details);
                logEntry.put("success", success);
                logEntry.put("service", "EthereumDataService");

                String logContent = objectMapper.writeValueAsString(logEntry);

                if (success) {
                    minIOService.archiveSyncLog("ETH", logContent);
                } else {
                    minIOService.archiveErrorLog(logContent);
                }

            } catch (Exception e) {
                log.error("归档同步日志失败", e);
            }
        });
    }

    // ============= 1. 分块加载历史数据 =============

    @Override
    @Transactional
    public ApiResponse<String> syncHistoricalData(LocalDate startDate, LocalDate endDate, Integer batchDays) {
        try {
            log.info("开始同步历史数据: {} 到 {}", startDate, endDate);
            archiveSyncLog("syncHistoricalData",
                    String.format("开始同步历史数据: %s 到 %s", startDate, endDate), true);

            // 参数验证
            if (startDate == null) startDate = LocalDate.of(2026, 1, 20);
            if (endDate == null) endDate = LocalDate.now().minusDays(1);
            if (batchDays == null || batchDays <= 0) batchDays = 1;
            if (batchDays > 30) batchDays = 30;

            int totalBlocks = 0;
            int totalTransactions = 0;

            LocalDate currentDate = startDate;
            while (!currentDate.isAfter(endDate)) {
                LocalDate batchEndDate = currentDate.plusDays(batchDays - 1);
                if (batchEndDate.isAfter(endDate)) {
                    batchEndDate = endDate;
                }

                log.info("同步日期批次: {} 到 {}", currentDate, batchEndDate);

                // 同步区块
                int blocks = syncBlocksByDateRange(currentDate, batchEndDate);
                totalBlocks += blocks;

                // 同步交易
                int transactions = syncTransactionsByDateRange(currentDate, batchEndDate);
                totalTransactions += transactions;

                currentDate = batchEndDate.plusDays(1);
                Thread.sleep(1000); // 避免请求过于频繁
            }

            archiveSyncLog("syncHistoricalData",
                    String.format("历史数据同步完成，共同步 %d 个区块，%d 笔交易",
                            totalBlocks, totalTransactions), true);

            return ApiResponse.success(String.format("历史数据同步完成，共同步 %d 个区块，%d 笔交易",
                    totalBlocks, totalTransactions), null);

        } catch (Exception e) {
            log.error("同步历史数据失败", e);
            archiveSyncLog("syncHistoricalData",
                    String.format("同步失败: %s", e.getMessage()), false);
            return ApiResponse.error(500, "历史数据同步失败: " + e.getMessage());
        }
    }

    private int syncBlocksByDateRange(LocalDate startDate, LocalDate endDate) {
        try {
            String query = String.format(
                    "SELECT `number`, `hash`, `parent_hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_ethereum.blocks` " +
                            "WHERE DATE(`timestamp`) BETWEEN '%s' AND '%s' " +
                            "ORDER BY `number` ASC ",
                    startDate.toString(), endDate.toString()
            );

            TableResult result = executeBigQuery(query);

            // 归档原始查询结果
            if (minioArchiveEnabled) {
                archiveBigQueryResult("blocks_query", result);
            }

            List<ChainBlock> blocks = mapToChainBlocks(result);

            int savedCount = 0;
            for (ChainBlock block : blocks) {
                try {
                    // 检查是否已存在
                    Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                    if (!existingBlock.isPresent()) {
                        block.setChain(CHAIN_ETH);
                        chainBlockRepository.save(block);

                        // 归档区块数据
                        archiveBlockData(block);

                        savedCount++;
                        log.debug("保存新区块: {}", block.getHeight());
                    }
                } catch (Exception e) {
                    log.error("保存区块失败: height={}", block.getHeight(), e);
                }
            }

            log.info("同步区块: {} 到 {}，新增 {} 个", startDate, endDate, savedCount);
            return savedCount;

        } catch (Exception e) {
            log.error("同步区块失败: {} - {}", startDate, endDate, e);
            return 0;
        }
    }

    // 修改 syncTransactionsByDateRange 方法，移除 transaction_index < 20 限制
    private int syncTransactionsByDateRange(LocalDate startDate, LocalDate endDate) {
        try {
            int totalSaved = 0;
            LocalDate currentDate = startDate;
            List<ChainTx> batchForGraph = new ArrayList<>(); // 批量保存到图数据库

            while (!currentDate.isAfter(endDate)) {
                // 移除 transaction_index < 20 限制
                String query = String.format(
                        "SELECT `hash`, `block_number`, `block_timestamp`, `from_address`, `to_address`, `value`, " +
                                "`gas_price`, `receipt_gas_used`, `gas`, `nonce`, `input`, `transaction_index`, " +
                                "`receipt_status` " +
                                "FROM `bigquery-public-data.crypto_ethereum.transactions` " +
                                "WHERE DATE(`block_timestamp`) = '%s' " +
                                // "  AND `transaction_index` < 20 " + // 移除这个限制
                                "ORDER BY `block_number` DESC, `transaction_index` " +
                                "LIMIT 1000", // 添加限制避免数据量过大
                        currentDate.toString()
                );

                TableResult result = executeBigQuery(query);

                // 归档原始查询结果
                if (minioArchiveEnabled) {
                    archiveBigQueryResult("transactions_query", result);
                }

                List<ChainTx> transactions = mapToChainTxs(result);

                int savedCount = 0;
                for (ChainTx tx : transactions) {
                    try {
                        Optional<ChainTx> existingTx = chainTxRepository.findByChainAndTxHash(CHAIN_ETH, tx.getTxHash());
                        if (!existingTx.isPresent()) {
                            tx.setChain(CHAIN_ETH);
                            ChainTx savedTx = chainTxRepository.save(tx);
                            savedCount++;

                            batchForGraph.add(savedTx);

                            // 归档交易数据
                            archiveTransactionData(savedTx);

                            // 保存输入输出
                            saveTransactionInputOutput(savedTx);
                        }
                    } catch (Exception e) {
                        log.error("保存交易失败: {}", tx.getTxHash().substring(0, 16), e);
                    }
                }

                // 图数据库保存 - 批量保存
                if (!batchForGraph.isEmpty() && graphService != null) {
                    try {
                        log.info("开始批量保存 {} 笔交易到图数据库", batchForGraph.size());
                        graphService.saveTransactionsToGraph(batchForGraph);
                        log.info("图数据库批量保存完成");
                        batchForGraph.clear(); // 清空批次
                    } catch (Exception e) {
                        log.error("批量保存到图数据库失败，尝试逐笔保存", e);
                        // 如果批量失败，尝试逐笔保存
                        for (ChainTx tx : batchForGraph) {
                            try {
                                graphService.saveTransactionToGraph(tx);
                            } catch (Exception ex) {
                                log.error("逐笔保存交易失败: {}", tx.getTxHash(), ex);
                            }
                        }
                        batchForGraph.clear();
                    }
                }

                totalSaved += savedCount;
                log.info("日期 {} 同步了 {} 笔交易", currentDate, savedCount);
                currentDate = currentDate.plusDays(1);
                Thread.sleep(500); // 避免请求过于频繁
            }

            return totalSaved;

        } catch (Exception e) {
            log.error("同步交易失败: {} - {}", startDate, endDate, e);
            return 0;
        }
    }

    // 添加一个新的方法来测试图数据库连接和保存
    // 简化测试方法
    @Override
    public ApiResponse<String> testGraphConnection() {
        try {
            if (graphService == null) {
                return ApiResponse.error(500, "GraphService未初始化");
            }

            log.info("测试图数据库连接...");
            return ApiResponse.success("图数据库服务已注入", null);

        } catch (Exception e) {
            log.error("测试图数据库连接失败", e);
            return ApiResponse.error(500, "测试图数据库连接失败: " + e.getMessage());
        }
    }

    private void saveTransactionInputOutput(ChainTx tx) {
        try {
            // 保存输入
            ChainTxInput input = new ChainTxInput();
            input.setChain(CHAIN_ETH);
            input.setTransaction(tx);
            input.setInputIndex(0);
            input.setAddress(tx.getFromAddress());
            input.setValue(tx.getTotalInput() != null ? tx.getTotalInput() : BigDecimal.ZERO);
            chainTxInputRepository.save(input);

            // 保存输出（如果to_address不为空）
            if (tx.getToAddress() != null && !tx.getToAddress().isEmpty()) {
                ChainTxOutput output = new ChainTxOutput();
                output.setChain(CHAIN_ETH);
                output.setTransaction(tx);
                output.setOutputIndex(0);
                output.setAddress(tx.getToAddress());
                output.setValue(tx.getTotalOutput() != null ? tx.getTotalOutput() : BigDecimal.ZERO);
                chainTxOutputRepository.save(output);
            }

        } catch (Exception e) {
            log.error("保存交易输入输出失败: txHash={}", tx.getTxHash(), e);
        }
    }

    // ============= 2. 获取最新数据 =============

    @Override
    @Transactional
    public ApiResponse<Map<String, Object>> getLatestData() {
        try {
            Map<String, Object> result = new HashMap<>();

            // 获取数据库最新数据
            Long latestHeightInDB = getLatestBlockHeightFromDB();
            LocalDateTime latestTimeInDB = getLatestBlockTimeFromDB();

            result.put("db_latest_height", latestHeightInDB);
            result.put("db_latest_time", latestTimeInDB);

            // 获取BigQuery最新区块高度
            String latestBlockQuery =
                    "SELECT MAX(`number`) as max_height, MAX(`timestamp`) as max_time " +
                            "FROM `bigquery-public-data.crypto_ethereum.blocks` " +
                            "WHERE DATE(`timestamp`) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)";

            TableResult bqResult = executeBigQuery(latestBlockQuery);

            // 归档查询结果
            if (minioArchiveEnabled) {
                archiveBigQueryResult("latest_block_query", bqResult);
            }

            for (FieldValueList row : bqResult.iterateAll()) {
                if (!row.get("max_height").isNull()) {
                    result.put("bq_latest_height", row.get("max_height").getLongValue());
                }
                if (!row.get("max_time").isNull()) {
                    String timestampStr = row.get("max_time").getStringValue();
                    LocalDateTime bqLatestTime = parseBigQueryTimestamp(timestampStr);
                    result.put("bq_latest_time", bqLatestTime);
                }
            }

            // 计算落后情况
            Long dbHeight = latestHeightInDB != null ? latestHeightInDB : 0L;
            Long bqHeight = (Long) result.getOrDefault("bq_latest_height", 0L);
            result.put("behind_blocks", bqHeight - dbHeight);

            if (latestTimeInDB != null && result.get("bq_latest_time") != null) {
                LocalDateTime bqTime = (LocalDateTime) result.get("bq_latest_time");
                long hoursBehind = java.time.Duration.between(latestTimeInDB, bqTime).toHours();
                result.put("behind_hours", hoursBehind);
            }

            return ApiResponse.success(result, null);

        } catch (Exception e) {
            log.error("获取最新数据失败", e);
            return ApiResponse.error(500, "获取最新数据失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public ApiResponse<String> syncLatestData(Integer blocksToSync) {
        try {
            if (blocksToSync == null || blocksToSync <= 0) blocksToSync = 100;
            if (blocksToSync > 1000) blocksToSync = 1000;

            Long latestHeightInDB = getLatestBlockHeightFromDB();
            if (latestHeightInDB == null) latestHeightInDB = 0L;

            log.info("开始同步最新数据，数据库最新高度: {}，同步 {} 个区块", latestHeightInDB, blocksToSync);
            archiveSyncLog("syncLatestData",
                    String.format("开始同步最新数据，数据库最新高度: %s，同步 %s 个区块", latestHeightInDB, blocksToSync), true);

            // 获取最新区块
            String blocksQuery = String.format(
                    "SELECT `number`, `hash`, `parent_hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_ethereum.blocks` " +
                            "WHERE `number` > %d " +
                            "  AND DATE(`timestamp`) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) " +
                            "ORDER BY `number` ASC " +
                            "LIMIT %d",
                    latestHeightInDB, blocksToSync
            );

            TableResult blocksResult = executeBigQuery(blocksQuery);

            // 归档查询结果
            if (minioArchiveEnabled) {
                archiveBigQueryResult("latest_blocks_query", blocksResult);
            }

            List<ChainBlock> blocks = mapToChainBlocks(blocksResult);

            if (blocks.isEmpty()) {
                return ApiResponse.success("已是最新数据，无需同步", null);
            }

            int savedBlocks = 0;
            int savedTransactions = 0;

            for (ChainBlock block : blocks) {
                try {
                    // 检查是否已存在
                    Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                    if (!existingBlock.isPresent()) {
                        block.setChain(CHAIN_ETH);
                        chainBlockRepository.save(block);

                        // 归档区块数据
                        archiveBlockData(block);

                        savedBlocks++;

                        // 同步该区块的交易
                        savedTransactions += syncBlockTransactions(block.getHeight());
                    }
                } catch (Exception e) {
                    log.error("保存区块失败: height={}", block.getHeight(), e);
                }
            }

            archiveSyncLog("syncLatestData",
                    String.format("同步完成，新增 %d 个区块，%d 笔交易", savedBlocks, savedTransactions), true);

            return ApiResponse.success(String.format("同步完成，新增 %d 个区块，%d 笔交易",
                    savedBlocks, savedTransactions), null);

        } catch (Exception e) {
            log.error("同步最新数据失败", e);
            archiveSyncLog("syncLatestData", String.format("同步失败: %s", e.getMessage()), false);
            return ApiResponse.error(500, "同步最新数据失败: " + e.getMessage());
        }
    }

    private int syncBlockTransactions(Long blockHeight) {
        try {
            String query = String.format(
                    "SELECT `hash`, `block_number`, `block_timestamp`, `from_address`, `to_address`, `value`, " +
                            "`gas_price`, `receipt_gas_used`, `gas`, `nonce`, `input`, `transaction_index`, " +
                            "`receipt_status` " +
                            "FROM `bigquery-public-data.crypto_ethereum.transactions` " +
                            "WHERE `block_number` = %d " +
                            "ORDER BY `transaction_index` ",
                    blockHeight
            );

            TableResult result = executeBigQuery(query);

            // 归档查询结果
            if (minioArchiveEnabled) {
                archiveBigQueryResult("block_transactions_query", result);
            }

            List<ChainTx> transactions = mapToChainTxs(result);

            List<ChainTx> batchForGraph = new ArrayList<>();

            int savedCount = 0;
            for (ChainTx tx : transactions) {
                try {
                    Optional<ChainTx> existingTx = chainTxRepository.findByChainAndTxHash(CHAIN_ETH, tx.getTxHash());
                    if (!existingTx.isPresent()) {
                        tx.setChain(CHAIN_ETH);
                        ChainTx savedTx = chainTxRepository.save(tx);
                        savedCount++;

                        batchForGraph.add(savedTx);

                        // 归档交易数据
                        archiveTransactionData(savedTx);

                        // 保存输入输出
                        saveTransactionInputOutput(savedTx);
                    }
                } catch (Exception e) {
                    log.error("保存交易失败: {}", tx.getTxHash().substring(0, 16), e);
                }
            }

            if (!batchForGraph.isEmpty() && graphService != null) {
                try {
                    graphService.saveTransactionsToGraph(batchForGraph);
                } catch (Exception e) {
                    log.error("批量保存到图数据库失败", e);
                }
            }

            log.debug("同步区块 {} 的 {} 笔交易", blockHeight, savedCount);
            return savedCount;

        } catch (Exception e) {
            log.error("同步区块交易失败: {}", blockHeight, e);
            return 0;
        }
    }

    // ============= 3. 定时同步 =============

    @Scheduled(fixedDelay = 300000) // 每5分钟执行一次
    @Transactional
    public void scheduledSyncLatestData() {
        try {
            log.info("开始定时同步最新数据");
            archiveSyncLog("scheduledSyncLatestData", "开始定时同步最新数据", true);

            Long latestHeightInDB = getLatestBlockHeightFromDB();
            if (latestHeightInDB == null) {
                log.info("数据库无数据，跳过定时同步");
                return;
            }

            // 只同步最新的20个区块
            String blocksQuery = String.format(
                    "SELECT `number`, `hash`, `parent_hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_ethereum.blocks` " +
                            "WHERE `number` > %d " +
                            "  AND DATE(`timestamp`) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) " +
                            "ORDER BY `number` ASC " +
                            "LIMIT 20",
                    latestHeightInDB
            );

            TableResult blocksResult = executeBigQuery(blocksQuery);

            // 归档查询结果
            if (minioArchiveEnabled) {
                archiveBigQueryResult("scheduled_blocks_query", blocksResult);
            }

            List<ChainBlock> blocks = mapToChainBlocks(blocksResult);

            if (!blocks.isEmpty()) {
                int savedBlocks = 0;
                for (ChainBlock block : blocks) {
                    Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                    if (!existingBlock.isPresent()) {
                        block.setChain(CHAIN_ETH);
                        chainBlockRepository.save(block);

                        // 归档区块数据
                        archiveBlockData(block);

                        savedBlocks++;

                        // 同步交易
                        syncBlockTransactions(block.getHeight());
                    }
                }
                log.info("定时同步完成，新增 {} 个区块", savedBlocks);
                archiveSyncLog("scheduledSyncLatestData",
                        String.format("定时同步完成，新增 %s 个区块", savedBlocks), true);
            } else {
                log.info("定时同步：无新数据");
            }

        } catch (Exception e) {
            log.error("定时同步失败", e);
            archiveSyncLog("scheduledSyncLatestData",
                    String.format("定时同步失败: %s", e.getMessage()), false);
        }
    }

    // ============= 4. 数据库查询功能 =============

    @Override
    @Cacheable(value = "blockchainStats", key = "'stats'")
    public ApiResponse<Map<String, Object>> getBlockchainStats() {
        try {
            Map<String, Object> stats = new HashMap<>();

            // 基础统计
            Long blockCount = chainBlockRepository.countByChain(CHAIN_ETH);
            Long txCount = chainTxRepository.countByChain(CHAIN_ETH);
            Long latestHeight = getLatestBlockHeightFromDB();
            LocalDateTime latestTime = getLatestBlockTimeFromDB();

            stats.put("chain", "Ethereum");
            stats.put("blockCount", blockCount != null ? blockCount : 0);
            stats.put("transactionCount", txCount != null ? txCount : 0);
            stats.put("latestBlockHeight", latestHeight != null ? latestHeight : 0);
            stats.put("latestBlockTime", latestTime);
            stats.put("timestamp", LocalDateTime.now());

            return ApiResponse.success(stats, null);

        } catch (Exception e) {
            log.error("获取区块链统计失败", e);
            return ApiResponse.error(500, "获取区块链统计失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Page<ChainBlock>> getBlocksPage(Integer page, Integer size) {
        try {
            if (page == null || page < 0) page = 0;
            if (size == null || size <= 0) size = 20;
            if (size > 100) size = 100;

            Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "height"));
            Page<ChainBlock> blocks = chainBlockRepository.findByChain(CHAIN_ETH, pageable);

            return ApiResponse.success(blocks, blocks.getTotalElements());

        } catch (Exception e) {
            log.error("分页查询区块失败", e);
            return ApiResponse.error(500, "分页查询区块失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Page<ChainTx>> getTransactionsPage(Integer page, Integer size) {
        try {
            if (page == null || page < 0) page = 0;
            if (size == null || size <= 0) size = 20;
            if (size > 100) size = 100;

            Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "blockHeight", "txIndex"));
            Page<ChainTx> transactions = chainTxRepository.findByChain(CHAIN_ETH, pageable);

            return ApiResponse.success(transactions, transactions.getTotalElements());

        } catch (Exception e) {
            log.error("分页查询交易失败", e);
            return ApiResponse.error(500, "分页查询交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainTx>> getTransactionsByAddress(String address, Integer limit) {
        try {
            if (limit == null || limit <= 0) limit = 20;
            if (limit > 100) limit = 100;

            Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "blockHeight", "txIndex"));

            // 查询地址相关的交易
            Page<ChainTx> transactions = chainTxRepository.findByFromAddressOrToAddress(CHAIN_ETH, address, pageable);

            return ApiResponse.success(transactions.getContent(), transactions.getTotalElements());

        } catch (Exception e) {
            log.error("查询地址交易失败: {}", address, e);
            return ApiResponse.error(500, "查询地址交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressBalance(String address) {
        try {
            Map<String, Object> result = new HashMap<>();

            // 计算地址余额
            BigDecimal balance = calculateAddressBalance(address);

            // 获取交易数量
            Long txCount = chainTxRepository.countByFromAddressOrToAddress(CHAIN_ETH, address);

            result.put("address", address);
            result.put("balance", balance);
            result.put("transactionCount", txCount != null ? txCount : 0);
            result.put("chain", CHAIN_ETH);

            return ApiResponse.success(result, null);

        } catch (Exception e) {
            log.error("查询地址余额失败: {}", address, e);
            return ApiResponse.error(500, "查询地址余额失败: " + e.getMessage());
        }
    }

    private BigDecimal calculateAddressBalance(String address) {
        try {
            // 查询地址作为接收者的总金额
            BigDecimal received = chainTxRepository.sumTotalOutputByToAddress(CHAIN_ETH, address);
            if (received == null) received = BigDecimal.ZERO;

            // 查询地址作为发送者的总金额（包括手续费）
            BigDecimal sent = chainTxRepository.sumTotalInputByFromAddress(CHAIN_ETH, address);
            if (sent == null) sent = BigDecimal.ZERO;

            // 余额 = 总收入 - 总支出
            return received.subtract(sent);

        } catch (Exception e) {
            log.error("计算地址余额失败: {}", address, e);
            return BigDecimal.ZERO;
        }
    }

    @Override
    public ApiResponse<List<Map<String, Object>>> getDailyStats(Integer days) {
        try {
            if (days == null || days <= 0) days = 7;
            if (days > 30) days = 30;

            List<Map<String, Object>> dailyStats = new ArrayList<>();
            LocalDate endDate = LocalDate.now();
            LocalDate startDate = endDate.minusDays(days);

            LocalDate currentDate = startDate;
            while (!currentDate.isAfter(endDate)) {
                Map<String, Object> dayStat = new HashMap<>();

                LocalDateTime dayStart = currentDate.atStartOfDay();
                LocalDateTime dayEnd = currentDate.plusDays(1).atStartOfDay();

                // 获取当天的区块数量
                Long blocksCount = chainBlockRepository.countByChainAndBlockTimeBetween(
                        CHAIN_ETH, dayStart, dayEnd);

                // 获取当天的交易数量
                Long transactionsCount = chainTxRepository.countByChainAndBlockTimeBetween(
                        CHAIN_ETH, dayStart, dayEnd);

                dayStat.put("date", currentDate.toString());
                dayStat.put("blocks", blocksCount != null ? blocksCount : 0);
                dayStat.put("transactions", transactionsCount != null ? transactionsCount : 0);

                dailyStats.add(dayStat);
                currentDate = currentDate.plusDays(1);
            }

            return ApiResponse.success(dailyStats, (long) dailyStats.size());

        } catch (Exception e) {
            log.error("获取每日统计失败", e);
            return ApiResponse.error(500, "获取每日统计失败: " + e.getMessage());
        }
    }

    // ============= 基础查询方法 =============

    @Override
    @Transactional
    public ApiResponse<List<ChainBlock>> getBlocks(Long startHeight, Long endHeight, Integer limit) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;

            // 从数据库查询
            List<ChainBlock> blocks = getBlocksFromDatabase(startHeight, endHeight, limit);

            if (!blocks.isEmpty()) {
                return ApiResponse.success(blocks, (long) blocks.size());
            }

            // 如果数据库没有，从BigQuery查询（添加时间限制）
            LocalDate sevenDaysAgo = LocalDate.now().minusDays(7);
            String query = String.format(
                    "SELECT `number`, `hash`, `parent_hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_ethereum.blocks` " +
                            "WHERE `number` >= %d AND `number` <= %d " +
                            "  AND DATE(`timestamp`) >= '%s' " +
                            "ORDER BY `number` DESC " +
                            "LIMIT %d",
                    startHeight != null ? startHeight : 0,
                    endHeight != null ? endHeight : Long.MAX_VALUE,
                    sevenDaysAgo.toString(),
                    limit
            );

            TableResult result = executeBigQuery(query);

            // 归档查询结果
            if (minioArchiveEnabled) {
                archiveBigQueryResult("get_blocks_query", result);
            }

            List<ChainBlock> blocksFromBQ = mapToChainBlocks(result);

            // 保存到数据库
            for (ChainBlock block : blocksFromBQ) {
                try {
                    Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                    if (!existingBlock.isPresent()) {
                        block.setChain(CHAIN_ETH);
                        chainBlockRepository.save(block);

                        // 归档区块数据
                        archiveBlockData(block);
                    }
                } catch (Exception e) {
                    log.error("保存区块失败: height={}", block.getHeight(), e);
                }
            }

            return ApiResponse.success(blocksFromBQ, (long) blocksFromBQ.size());

        } catch (Exception e) {
            log.error("获取区块数据失败", e);
            return ApiResponse.error(500, "获取区块数据失败: " + e.getMessage());
        }
    }

    private List<ChainBlock> getBlocksFromDatabase(Long startHeight, Long endHeight, Integer limit) {
        Pageable pageable = PageRequest.of(0, limit, Sort.by("height").descending());

        if (startHeight != null && endHeight != null) {
            return chainBlockRepository.findByChainAndHeightBetween(CHAIN_ETH, startHeight, endHeight, pageable);
        } else if (startHeight != null) {
            return chainBlockRepository.findByChainAndHeightGreaterThanEqual(CHAIN_ETH, startHeight, pageable);
        } else if (endHeight != null) {
            return chainBlockRepository.findByChainAndHeightLessThanEqual(CHAIN_ETH, endHeight, pageable);
        } else {
            return chainBlockRepository.findAllByChain(CHAIN_ETH, pageable);
        }
    }

    @Override
    @Transactional
    public ApiResponse<List<ChainTx>> getTransactions(Long blockHeight, Integer limit, Integer offset) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;
            if (offset == null || offset < 0) offset = 0;

            // 从数据库查询
            Pageable pageable = PageRequest.of(offset / limit, limit, Sort.by("txIndex").ascending());
            List<ChainTx> transactions = chainTxRepository.findByChainAndBlockHeight(CHAIN_ETH, blockHeight, pageable);

            if (!transactions.isEmpty()) {
                return ApiResponse.success(transactions, (long) transactions.size());
            }

            // 从BigQuery查询（添加时间限制）
            LocalDate sevenDaysAgo = LocalDate.now().minusDays(7);
            String query = String.format(
                    "SELECT `hash`, `block_number`, `block_timestamp`, `from_address`, `to_address`, `value`, " +
                            "`gas_price`, `receipt_gas_used`, `gas`, `nonce`, `input`, `transaction_index`, " +
                            "`receipt_status` " +
                            "FROM `bigquery-public-data.crypto_ethereum.transactions` " +
                            "WHERE `block_number` = %d " +
                            "  AND DATE(`block_timestamp`) >= '%s' " +
                            "ORDER BY `transaction_index` " +
                            "LIMIT %d OFFSET %d",
                    blockHeight, sevenDaysAgo.toString(), limit, offset
            );

            TableResult result = executeBigQuery(query);

            // 归档查询结果
            if (minioArchiveEnabled) {
                archiveBigQueryResult("get_transactions_query", result);
            }

            List<ChainTx> transactionsFromBQ = mapToChainTxs(result);

            // 保存到数据库
            for (ChainTx tx : transactionsFromBQ) {
                try {
                    Optional<ChainTx> existingTx = chainTxRepository.findByChainAndTxHash(CHAIN_ETH, tx.getTxHash());
                    if (!existingTx.isPresent()) {
                        tx.setChain(CHAIN_ETH);
                        ChainTx savedTx = chainTxRepository.save(tx);

                        // 归档交易数据
                        archiveTransactionData(savedTx);

                        saveTransactionInputOutput(savedTx);
                    }
                } catch (Exception e) {
                    log.error("保存交易失败: {}", tx.getTxHash().substring(0, 16), e);
                }
            }

            return ApiResponse.success(transactionsFromBQ, (long) transactionsFromBQ.size());

        } catch (Exception e) {
            log.error("获取交易数据失败", e);
            return ApiResponse.error(500, "获取交易数据失败: " + e.getMessage());
        }
    }

    // ============= 数据映射方法 =============

    private List<ChainBlock> mapToChainBlocks(TableResult result) {
        List<ChainBlock> blocks = new ArrayList<>();

        for (FieldValueList row : result.iterateAll()) {
            try {
                ChainBlock block = new ChainBlock();
                block.setHeight(row.get("number").getLongValue());
                block.setBlockHash(row.get("hash").getStringValue());

                if (!row.get("parent_hash").isNull()) {
                    block.setPrevBlockHash(row.get("parent_hash").getStringValue());
                }

                String timestampStr = row.get("timestamp").getStringValue();
                LocalDateTime blockTime = parseBigQueryTimestamp(timestampStr);
                block.setBlockTime(blockTime);

                block.setTxCount((int) row.get("transaction_count").getLongValue());
                block.setRawSizeBytes(row.get("size").getLongValue());

                blocks.add(block);
            } catch (Exception e) {
                log.error("映射区块数据失败", e);
            }
        }

        return blocks;
    }

    private List<ChainTx> mapToChainTxs(TableResult result) {
        List<ChainTx> transactions = new ArrayList<>();

        for (FieldValueList row : result.iterateAll()) {
            try {
                ChainTx tx = new ChainTx();
                tx.setTxHash(row.get("hash").getStringValue());
                tx.setBlockHeight(row.get("block_number").getLongValue());

                String timestampStr = row.get("block_timestamp").getStringValue();
                LocalDateTime blockTime = parseBigQueryTimestamp(timestampStr);
                tx.setBlockTime(blockTime);

                tx.setFromAddress(row.get("from_address").getStringValue());
                if (!row.get("to_address").isNull()) {
                    tx.setToAddress(row.get("to_address").getStringValue());
                }

                // 以太坊金额（单位转换：wei -> ETH）
                BigDecimal valueWei = BigDecimal.valueOf(row.get("value").getDoubleValue());
                BigDecimal valueEth = convertWeiToEth(valueWei);
                tx.setTotalOutput(valueEth);

                // 交易索引
                if (!row.get("transaction_index").isNull()) {
                    tx.setTxIndex((int) row.get("transaction_index").getLongValue());
                } else {
                    tx.setTxIndex(0);
                }

                // 交易状态
                if (!row.get("receipt_status").isNull()) {
                    long status = row.get("receipt_status").getLongValue();
                    tx.setStatus(status == 1 ? "confirmed" : "failed");
                } else {
                    tx.setStatus("confirmed");
                }

                // 计算手续费和总输入
                if (!row.get("gas_price").isNull() && !row.get("receipt_gas_used").isNull()) {
                    BigDecimal gasPriceWei = BigDecimal.valueOf(row.get("gas_price").getDoubleValue());
                    BigDecimal gasUsed = BigDecimal.valueOf(row.get("receipt_gas_used").getDoubleValue());
                    BigDecimal feeWei = gasPriceWei.multiply(gasUsed);
                    BigDecimal feeEth = convertWeiToEth(feeWei);
                    tx.setFee(feeEth);
                    tx.setTotalInput(valueEth.add(feeEth)); // 总输入 = 转账金额 + 手续费
                }

                transactions.add(tx);
            } catch (Exception e) {
                log.error("映射交易数据失败", e);
            }
        }

        return transactions;
    }

    // ============= 工具方法 =============

    private TableResult executeBigQuery(String query) throws InterruptedException {
        try {
            log.debug("执行 BigQuery 查询: {}", query);
            QueryJobConfiguration queryConfig = QueryJobConfiguration.newBuilder(query).build();
            return bigQuery.query(queryConfig);
        } catch (Exception e) {
            log.error("BigQuery 查询失败 - SQL: {}", query);
            throw e;
        }
    }

    private Long getLatestBlockHeightFromDB() {
        return chainBlockRepository.findMaxHeight(CHAIN_ETH);
    }

    private LocalDateTime getLatestBlockTimeFromDB() {
        return chainBlockRepository.findLatestBlockTime(CHAIN_ETH);
    }

    private BigDecimal convertWeiToEth(BigDecimal wei) {
        // 1 ETH = 10^18 wei
        return wei.divide(BigDecimal.valueOf(1_000_000_000_000_000_000L), 18, BigDecimal.ROUND_HALF_UP);
    }

    private LocalDateTime parseBigQueryTimestamp(String timestampStr) {
        try {
            // BigQuery 返回的时间戳可能是 Unix 时间戳（数字）或字符串
            if (timestampStr.matches("-?\\d+(\\.\\d+)?([Ee][+-]?\\d+)?")) {
                double timestampDouble = Double.parseDouble(timestampStr);
                long milliseconds = (long) (timestampDouble * 1000);
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(milliseconds), ZoneId.of("UTC"));
            } else {
                String cleaned = timestampStr.replace(" UTC", "").replace("+00:00", "").trim();

                // 尝试多种格式
                String[] formats = {
                        "yyyy-MM-dd HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss.SSS",
                        "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
                };

                for (String format : formats) {
                    try {
                        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
                        return LocalDateTime.parse(cleaned, formatter);
                    } catch (Exception e) {
                        // 继续尝试下一种格式
                    }
                }

                return LocalDateTime.parse(cleaned, DateTimeFormatter.ISO_DATE_TIME);
            }
        } catch (Exception e) {
            log.warn("解析时间戳失败: {}, 使用当前时间", timestampStr);
            return LocalDateTime.now();
        }
    }

    // ============= 清理缓存方法 =============

    @Override
    @CacheEvict(value = {"blockchainStats", "blocksByTime", "transactionsByTime",
            "transactionDetail", "addressInfo"}, allEntries = true)
    public void clearAllCache() {
        log.info("清除所有缓存");
    }

    // ============= 实现缺少的接口方法 =============

    @Override
    public ApiResponse<Long> getBlockNumber() {
        try {
            Long latestHeight = getLatestBlockHeightFromDB();
            return ApiResponse.success(latestHeight, null);
        } catch (Exception e) {
            log.error("获取区块高度失败", e);
            return ApiResponse.error(500, "获取区块高度失败");
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getTransactionDetail(String txHash) {
        try {
            Optional<ChainTx> txOptional = chainTxRepository.findByChainAndTxHash(CHAIN_ETH, txHash);

            if (txOptional.isPresent()) {
                ChainTx tx = txOptional.get();
                Map<String, Object> result = new HashMap<>();
                result.put("transaction", tx);
                result.put("fromAddress", tx.getFromAddress());
                result.put("toAddress", tx.getToAddress());
                result.put("value", tx.getTotalOutput());
                result.put("fee", tx.getFee());
                result.put("blockHeight", tx.getBlockHeight());
                result.put("blockTime", tx.getBlockTime());

                return ApiResponse.success(result, null);
            } else {
                return ApiResponse.error(404, "交易不存在: " + txHash);
            }

        } catch (Exception e) {
            log.error("获取交易详情失败", e);
            return ApiResponse.error(500, "获取交易详情失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressInfo(String address) {
        return getAddressBalance(address);
    }

    @Override
    public ApiResponse<List<ChainTx>> getTransactionsByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;

            Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "blockHeight", "txIndex"));
            List<ChainTx> transactions = chainTxRepository.findByChainAndBlockTimeBetween(CHAIN_ETH, startTime, endTime, pageable);

            return ApiResponse.success(transactions, (long) transactions.size());

        } catch (Exception e) {
            log.error("按时间获取交易失败", e);
            return ApiResponse.error(500, "按时间获取交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainBlock>> getBlocksByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;

            Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "height"));
            List<ChainBlock> blocks = chainBlockRepository.findByChainAndBlockTimeBetween(CHAIN_ETH, startTime, endTime, pageable);

            return ApiResponse.success(blocks, (long) blocks.size());

        } catch (Exception e) {
            log.error("按时间获取区块失败", e);
            return ApiResponse.error(500, "按时间获取区块失败: " + e.getMessage());
        }
    }

    // ============= 其他接口方法 =============

    @Override
    public ApiResponse<String> exportBlocksToCsv(Long startHeight, Long endHeight) {
        return ApiResponse.success("从数据库导出功能待实现", null);
    }

    @Override
    public ApiResponse<String> exportTransactionsToCsv(LocalDateTime startTime, LocalDateTime endTime) {
        return ApiResponse.success("从数据库导出功能待实现", null);
    }


}