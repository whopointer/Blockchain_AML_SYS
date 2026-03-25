// com/seecoder/DataProcessing/serviceImpl/EthereumDataServiceImpl.java
package com.seecoder.DataProcessing.serviceImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.cloud.bigquery.*;
import com.opencsv.CSVWriter;
import com.seecoder.DataProcessing.po.*;
import com.seecoder.DataProcessing.repository.*;
import com.seecoder.DataProcessing.repository.clickhouse.ClickHouseStatsRepository;
import com.seecoder.DataProcessing.service.ClickHouseAggregationService;
import com.seecoder.DataProcessing.service.EthereumDataService;
import com.seecoder.DataProcessing.service.GraphService;
import com.seecoder.DataProcessing.service.MinIOService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import com.seecoder.DataProcessing.vo.ExploreTaskStatus;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

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

    @Autowired
    private ClickHouseStatsRepository clickHouseStatsRepository;

    @Autowired
    private ClickHouseAggregationService clickHouseAggregationService;

    @Autowired
    private ChainTokenTransferRepository chainTokenTransferRepository;

    private final Map<String, ExploreTaskStatus> taskStatusMap = new ConcurrentHashMap<>();

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

    @Scheduled(cron = "0 0 4 * * ?") // 每5分钟执行一次
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

                // 处理 value 字段（wei），安全转换为 BigInteger
                FieldValue valueField = row.get("value");
                BigInteger valueWei;
                if (!valueField.isNull()) {
                    if (valueField.getAttribute() == FieldValue.Attribute.PRIMITIVE) {
                        // 尝试获取长整型（适用于较小数值）
                        try {
                            valueWei = BigInteger.valueOf(valueField.getLongValue());
                        } catch (Exception e) {
                            // 如果溢出或类型不匹配，使用字符串
                            valueWei = new BigInteger(valueField.getStringValue());
                        }
                    } else {
                        // 其他类型（如 NUMERIC）直接使用字符串
                        valueWei = new BigInteger(valueField.getStringValue());
                    }
                } else {
                    valueWei = BigInteger.ZERO;
                }
                tx.setValueWei(valueWei);
                BigDecimal valueEth = convertWeiToEth(new BigDecimal(valueWei));
                tx.setTotalOutput(valueEth);

                // 交易哈希
                tx.setTxHash(row.get("hash").getStringValue());

                // 区块号
                tx.setBlockHeight(row.get("block_number").getLongValue());

                // 区块时间戳
                String timestampStr = row.get("block_timestamp").getStringValue();
                LocalDateTime blockTime = parseBigQueryTimestamp(timestampStr);
                tx.setBlockTime(blockTime);

                // 发送地址
                tx.setFromAddress(row.get("from_address").getStringValue());

                // 接收地址（可能为空）
                if (!row.get("to_address").isNull()) {
                    tx.setToAddress(row.get("to_address").getStringValue());
                }

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
                    // gas_price 和 receipt_gas_used 可能返回 NUMERIC 类型，使用字符串转换
                    FieldValue gasPriceField = row.get("gas_price");
                    FieldValue gasUsedField = row.get("receipt_gas_used");
                    BigInteger gasPriceWei = new BigInteger(gasPriceField.getStringValue());
                    BigInteger gasUsed = new BigInteger(gasUsedField.getStringValue());
                    BigInteger feeWei = gasPriceWei.multiply(gasUsed);
                    BigDecimal feeEth = convertWeiToEth(new BigDecimal(feeWei));
                    tx.setFee(feeEth);
                    tx.setTotalInput(valueEth.add(feeEth)); // 总输入 = 转账金额 + 手续费
                }

                // 交易大小（字节）
                // 以太坊交易在 BigQuery 中无 size 字段，默认设为 0
                tx.setSizeBytes(0L);

                // 以太坊无 locktime，默认设为 0
                tx.setLocktime(0L);

                transactions.add(tx);
            } catch (Exception e) {
                log.error("映射交易数据失败", e);
            }
        }

        return transactions;
    }



    // ============= 新增：异步地址探索方法 =============

    @Override
    @Async
    public CompletableFuture<ApiResponse<List<String>>> exploreAndExport(String taskId,
                                                                         List<String> sources,
                                                                         List<String> allowed,
                                                                         List<String> forbidden,
                                                                         LocalDateTime startTime,
                                                                         LocalDateTime endTime) {
        log.info("开始地址探索任务 [{}]：源地址={}, 允许列表={}, 禁止列表={}, 时间范围={} - {}",
                taskId, sources, allowed, forbidden, startTime, endTime);

        // 初始化任务状态
        ExploreTaskStatus status = new ExploreTaskStatus();
        status.setTaskId(taskId);
        status.setStartTime(LocalDateTime.now());
        status.setStatus("RUNNING");
        taskStatusMap.put(taskId, status);

        try {
            // 1. 地址探索，收集所有涉及的交易哈希
            Set<String> allTxHashes = exploreAddressNetwork(sources, allowed, forbidden, startTime, endTime, status);
            log.info("任务 [{}] 探索完成，共收集到 {} 笔交易哈希", taskId, allTxHashes.size());

            // 2. 根据交易哈希获取完整的交易详情（从本地或BigQuery）
            List<ChainTx> fullTxs = fetchTransactionsByHashes(allTxHashes, startTime, endTime);
            log.info("任务 [{}] 获取到 {} 笔完整原生交易", taskId, fullTxs.size());

            // 3. 获取这些交易对应的代币转账事件
            List<ChainTokenTransfer> tokenTransfers = fetchTokenTransfersByTxHashes(allTxHashes, startTime, endTime);
            log.info("任务 [{}] 获取到 {} 条代币转账事件", taskId, tokenTransfers.size());

            // 4. 按区块范围分组（每100k个区块一组）
            Map<Long, List<ChainTx>> nativeGroup = new HashMap<>();
            Map<Long, List<ChainTokenTransfer>> tokenGroup = new HashMap<>();

            for (ChainTx tx : fullTxs) {
                long rangeStart = (tx.getBlockHeight() / 100000) * 100000;
                nativeGroup.computeIfAbsent(rangeStart, k -> new ArrayList<>()).add(tx);
            }
            for (ChainTokenTransfer tt : tokenTransfers) {
                long rangeStart = (tt.getBlockNumber() / 100000) * 100000;
                tokenGroup.computeIfAbsent(rangeStart, k -> new ArrayList<>()).add(tt);
            }

            // 5. 合并所有出现的区块范围
            Set<Long> allRanges = new HashSet<>();
            allRanges.addAll(nativeGroup.keySet());
            allRanges.addAll(tokenGroup.keySet());

            List<String> generatedFiles = new ArrayList<>();
            for (Long rangeStart : allRanges) {
                long rangeEnd = rangeStart + 100000;
                List<ChainTx> nativeList = nativeGroup.getOrDefault(rangeStart, Collections.emptyList());
                List<ChainTokenTransfer> tokenList = tokenGroup.getOrDefault(rangeStart, Collections.emptyList());

                // 生成CSV文件并上传到MinIO
                List<String> files = generateAndUploadCsv(rangeStart, rangeEnd, nativeList, tokenList);
                generatedFiles.addAll(files);

                // 更新任务进度
                status.setProgress((int) (generatedFiles.size() * 100 / allRanges.size()));
            }

            // 任务完成
            status.setStatus("COMPLETED");
            status.setEndTime(LocalDateTime.now());
            status.setResult(generatedFiles);
            status.setMessage("成功生成 " + generatedFiles.size() + " 个CSV文件");

            // 注意：ApiResponse.success 有两个重载：success(T data) 和 success(T data, Long total)
            // 这里我们使用 success(data)，因为 total 不是必需的
            return CompletableFuture.completedFuture(ApiResponse.success(generatedFiles, (long) generatedFiles.size()));
        } catch (Exception e) {
            log.error("任务 [{}] 执行失败", taskId, e);
            status.setStatus("FAILED");
            status.setEndTime(LocalDateTime.now());
            status.setMessage("失败：" + e.getMessage());
            return CompletableFuture.completedFuture(ApiResponse.error(500, "探索任务失败: " + e.getMessage()));
        }
    }

    // ============= 辅助方法：地址探索网络 =============

    private Set<String> exploreAddressNetwork(List<String> sources,
                                              List<String> allowed,
                                              List<String> forbidden,
                                              LocalDateTime startTime,
                                              LocalDateTime endTime,
                                              ExploreTaskStatus status) {
        Set<String> toExplore = new HashSet<>(sources);
        Set<String> explored = new HashSet<>();
        Set<String> forbiddenSet = new HashSet<>(forbidden);
        Set<String> allowedSet = new HashSet<>(allowed);
        Set<String> allTxHashes = new HashSet<>();

        while (!toExplore.isEmpty()) {
            String addr = toExplore.iterator().next();
            toExplore.remove(addr);
            if (explored.contains(addr)) continue;
            explored.add(addr);
            status.setProcessedAddresses(explored.size());

            // 1. 优先从Neo4j获取邻居地址和交易哈希
            Set<String> neighbors = getNeighborsFromGraph(addr, startTime, endTime);
            Set<String> txHashes = getTransactionHashesFromGraph(addr, startTime, endTime);

            // 2. 如果Neo4j数据不完整，回退到数据库/BigQuery
            if (neighbors.isEmpty() && txHashes.isEmpty()) {
                // 从本地数据库或BigQuery获取该地址的交易，提取对手地址和哈希
                List<ChainTx> txs = fetchTransactionsByAddressAndTime(addr, startTime, endTime);
                neighbors = new HashSet<>();
                txHashes = new HashSet<>();
                for (ChainTx tx : txs) {
                    txHashes.add(tx.getTxHash());
                    String counterparty = tx.getFromAddress().equals(addr) ? tx.getToAddress() : tx.getFromAddress();
                    if (counterparty != null) {
                        neighbors.add(counterparty);
                    }
                }
            }

            // 收集交易哈希
            allTxHashes.addAll(txHashes);

            // 根据规则决定是否继续探索邻居
            for (String neighbor : neighbors) {
                if (explored.contains(neighbor) || toExplore.contains(neighbor)) continue;
                boolean isForbidden = forbiddenSet.contains(neighbor);
                boolean isAllowed = allowedSet.contains(neighbor);
                if (!isForbidden || isAllowed) {
                    toExplore.add(neighbor);
                }
            }

            if (explored.size() % 100 == 0) {
                log.info("探索进度：已探索 {} 个地址，累计收集 {} 个交易哈希", explored.size(), allTxHashes.size());
            }
        }

        log.info("地址探索完成：共探索 {} 个地址，收集 {} 个交易哈希", explored.size(), allTxHashes.size());
        return allTxHashes;
    }

    // ============= 与 Neo4j 交互的辅助方法 =============

    private Set<String> getNeighborsFromGraph(String address, LocalDateTime start, LocalDateTime end) {
        try {
            return graphService.getNeighborAddresses(address, start, end);
        } catch (Exception e) {
            log.warn("从Neo4j获取邻居地址失败，将回退到数据库: {}", e.getMessage());
            return Collections.emptySet();
        }
    }

    private Set<String> getTransactionHashesFromGraph(String address, LocalDateTime start, LocalDateTime end) {
        try {
            return graphService.getTransactionHashes(address, start, end);
        } catch (Exception e) {
            log.warn("从Neo4j获取交易哈希失败，将回退到数据库: {}", e.getMessage());
            return Collections.emptySet();
        }
    }

    // ============= 从数据库获取地址交易 =============

    private List<ChainTx> fetchTransactionsByAddressAndTime(String address, LocalDateTime start, LocalDateTime end) {
        // 1. 从本地数据库查询
        Sort sort = Sort.by(Sort.Direction.ASC, "blockHeight", "txIndex");
        List<ChainTx> localTxs = chainTxRepository.findByAddressAndTimeRange(CHAIN_ETH, address, start, end, sort);

        log.info("从本地数据库查询到地址 {} 的交易数：{}", address, localTxs.size());
        return localTxs;
/*
        // 判断本地数据是否完整（简单规则：如果本地交易数量较少或时间范围较新，可能不全）
        long localCount = chainTxRepository.countByAddressAndTimeRange(CHAIN_ETH, address, start, end);
        long expectedMin = (end.toLocalDate().toEpochDay() - start.toLocalDate().toEpochDay()) * 5000; // 粗略估计
        if (localCount >= expectedMin) {
            return localTxs;
        }

        // 2. 从BigQuery拉取
        List<ChainTx> bqTxs = fetchTransactionsFromBigQuery(address, start, end);

        // 3. 保存新交易到数据库
        Set<String> localHashes = localTxs.stream().map(ChainTx::getTxHash).collect(Collectors.toSet());
        List<ChainTx> toSave = new ArrayList<>();
        for (ChainTx tx : bqTxs) {
            if (!localHashes.contains(tx.getTxHash())) {
                toSave.add(tx);
            }
        }
        if (!toSave.isEmpty()) {
            chainTxRepository.saveAll(toSave);
            // 异步归档到Neo4j
            graphService.saveTransactionsToGraph(toSave);
        }

        // 合并结果并排序
        List<ChainTx> merged = new ArrayList<>(localTxs);
        merged.addAll(toSave);
        merged.sort(Comparator.comparing(ChainTx::getBlockHeight).thenComparing(ChainTx::getTxIndex));
        return merged;

 */
    }

    private List<ChainTx> fetchTransactionsFromBigQuery(String address, LocalDateTime start, LocalDateTime end) {
        String query = String.format(
                "SELECT hash, block_number, block_timestamp, from_address, to_address, value, " +
                        "gas_price, receipt_gas_used, gas, nonce, input, transaction_index, receipt_status " +
                        "FROM `bigquery-public-data.crypto_ethereum.transactions` " +
                        "WHERE (from_address = '%s' OR to_address = '%s') " +
                        "  AND block_timestamp BETWEEN TIMESTAMP('%s') AND TIMESTAMP('%s')",
                address, address, start.toString(), end.toString()
        );
        try {
            TableResult result = executeBigQuery(query);
            archiveBigQueryResult("address_txs_query", result);
            return mapToChainTxs(result);
        } catch (Exception e) {
            log.error("从BigQuery拉取交易失败 address={}", address, e);
            return Collections.emptyList();
        }
    }

    // ============= 根据哈希列表获取完整交易 =============

    private List<ChainTx> fetchTransactionsByHashes(Set<String> txHashes, LocalDateTime start, LocalDateTime end) {
        if (txHashes.isEmpty()) return Collections.emptyList();

        // 1. 从本地数据库查询已存在的
        List<ChainTx> localTxs = chainTxRepository.findByChainAndTxHashIn(CHAIN_ETH, new ArrayList<>(txHashes));

        Set<String> localHashSet = localTxs.stream().map(ChainTx::getTxHash).collect(Collectors.toSet());
        Set<String> missingHashes = new HashSet<>(txHashes);
        missingHashes.removeAll(localHashSet);

        if (missingHashes.isEmpty()) {
            return localTxs;
        }
        return localTxs;


    }

    // ============= 获取代币转账事件 =============

    private List<ChainTokenTransfer> fetchTokenTransfersByTxHashes(Set<String> txHashes, LocalDateTime start, LocalDateTime end) {
        if (txHashes.isEmpty()) return Collections.emptyList();

        // 1. 从本地数据库查询
        List<ChainTokenTransfer> localTTs = chainTokenTransferRepository.findByTransactionHashIn(new ArrayList<>(txHashes));

        Set<String> localHashSet = localTTs.stream().map(ChainTokenTransfer::getTransactionHash).collect(Collectors.toSet());
        Set<String> missingHashes = new HashSet<>(txHashes);
        missingHashes.removeAll(localHashSet);

        if (missingHashes.isEmpty()) {
            return localTTs;
        }

        // 2. 从BigQuery分批拉取
        List<ChainTokenTransfer> bqTTs = new ArrayList<>();
        List<String> missingList = new ArrayList<>(missingHashes);
        int batchSize = 500;
        for (int i = 0; i < missingList.size(); i += batchSize) {
            List<String> batch = missingList.subList(i, Math.min(i + batchSize, missingList.size()));
            String inClause = String.join("','", batch);
            String query = String.format(
                    "SELECT block_number, block_timestamp, transaction_hash, log_index, token_address, from_address, to_address, value " +
                            "FROM `bigquery-public-data.crypto_ethereum.token_transfers` " +
                            "WHERE transaction_hash IN ('%s') " +
                            "  AND block_timestamp BETWEEN TIMESTAMP('%s') AND TIMESTAMP('%s')",
                    inClause, start.toString(), end.toString()
            );
            try {
                TableResult result = executeBigQuery(query);
                bqTTs.addAll(mapToChainTokenTransfers(result));
            } catch (Exception e) {
                log.error("批量拉取代币转账失败", e);
            }
        }

        // 3. 保存到本地数据库
        if (!bqTTs.isEmpty()) {
            chainTokenTransferRepository.saveAll(bqTTs);
        }

        List<ChainTokenTransfer> allTTs = new ArrayList<>(localTTs);
        allTTs.addAll(bqTTs);
        return allTTs;
    }

    /**
     * 将BigQuery结果映射为ChainTokenTransfer列表
     */
    private List<ChainTokenTransfer> mapToChainTokenTransfers(TableResult result) {
        List<ChainTokenTransfer> list = new ArrayList<>();
        for (FieldValueList row : result.iterateAll()) {
            try {
                ChainTokenTransfer tt = new ChainTokenTransfer();
                tt.setBlockNumber(row.get("block_number").getLongValue());
                tt.setBlockTimestamp(parseBigQueryTimestamp(row.get("block_timestamp").getStringValue()));
                tt.setTransactionHash(row.get("transaction_hash").getStringValue());
                tt.setLogIndex((int) row.get("log_index").getLongValue());
                tt.setTokenAddress(row.get("token_address").getStringValue());
                tt.setFromAddress(row.get("from_address").getStringValue());
                tt.setToAddress(row.get("to_address").getStringValue());

                // value 字段处理
                FieldValue valueField = row.get("value");
                BigInteger value;
                if (!valueField.isNull()) {
                    if (valueField.getAttribute() == FieldValue.Attribute.PRIMITIVE) {
                        try {
                            value = BigInteger.valueOf(valueField.getLongValue());
                        } catch (Exception e) {
                            value = new BigInteger(valueField.getStringValue());
                        }
                    } else {
                        value = new BigInteger(valueField.getStringValue());
                    }
                } else {
                    value = BigInteger.ZERO;
                }
                tt.setValue(value);

                list.add(tt);
            } catch (Exception e) {
                log.error("映射代币转账失败", e);
            }
        }
        return list;
    }

    // ============= 生成CSV文件并上传到MinIO =============

    private List<String> generateAndUploadCsv(long startBlock, long endBlock,
                                              List<ChainTx> nativeTxs,
                                              List<ChainTokenTransfer> tokenTxs) throws IOException {
        List<String> files = new ArrayList<>();

        // 为代币转账准备交易索引映射
        Map<String, Integer> txIndexMap = new HashMap<>();
        for (ChainTx tx : nativeTxs) {
            txIndexMap.put(tx.getTxHash(), tx.getTxIndex());
        }

        // 生成原生ETH CSV
        String nativeFileName = String.format("native_%d_%d.csv", startBlock, endBlock);
        try (CSVWriter writer = new CSVWriter(new FileWriter(nativeFileName))) {
            writer.writeNext(new String[]{"block", "timestamp", "index", "hex_tx", "coin", "from_addr", "to_addr", "value"});
            for (ChainTx tx : nativeTxs) {
                String[] row = {
                        String.valueOf(tx.getBlockHeight()),
                        formatTimestamp(tx.getBlockTime()),
                        String.valueOf(tx.getTxIndex() != null ? tx.getTxIndex() : 0),
                        tx.getTxHash(),
                        "0x0000000000000000000000000000000000000000",
                        tx.getFromAddress(),
                        tx.getToAddress() != null ? tx.getToAddress() : "0x0000000000000000000000000000000000000000",
                        tx.getValueWei() != null ? tx.getValueWei().toString() : "0"
                };
                writer.writeNext(row);
            }
        }
        files.add(nativeFileName);

        // 生成代币转账 CSV
        String tokenFileName = String.format("token_%d_%d.csv", startBlock, endBlock);
        try (CSVWriter writer = new CSVWriter(new FileWriter(tokenFileName))) {
            writer.writeNext(new String[]{"block", "timestamp", "index", "hex_tx", "coin", "from_addr", "to_addr", "value"});
            for (ChainTokenTransfer tt : tokenTxs) {
                Integer txIndex = txIndexMap.get(tt.getTransactionHash());
                String[] row = {
                        String.valueOf(tt.getBlockNumber()),
                        formatTimestamp(tt.getBlockTimestamp()),
                        String.valueOf(txIndex != null ? txIndex : 0),
                        tt.getTransactionHash(),
                        tt.getTokenAddress(),
                        tt.getFromAddress(),
                        tt.getToAddress(),
                        tt.getValue().toString()
                };
                writer.writeNext(row);
            }
        }
        files.add(tokenFileName);

        // 上传到MinIO
        for (String file : files) {
            minIOService.uploadFile(file);
        }

        // 删除本地临时文件
        for (String file : files) {
            new File(file).delete();
        }

        return files;
    }

    /**
     * 格式化时间为 CSV 要求的格式（yyyy-MM-dd HH:mm:ss）
     */
    private String formatTimestamp(LocalDateTime time) {
        return time.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
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

            Page<ChainTx> txPage = chainTxRepository.findByChainAndBlockTimeBetween(CHAIN_ETH, startTime, endTime, pageable);
            List<ChainTx> txs = txPage.getContent();
            return ApiResponse.success(txs, (long) txs.size());

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