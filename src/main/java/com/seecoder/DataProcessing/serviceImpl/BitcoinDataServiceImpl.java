// BitcoinDataServiceImpl.java
package com.seecoder.DataProcessing.serviceImpl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.cloud.bigquery.*;
import com.seecoder.DataProcessing.po.*;
import com.seecoder.DataProcessing.repository.*;
import com.seecoder.DataProcessing.service.BitcoinDataService;
import com.seecoder.DataProcessing.service.GraphService;
import com.seecoder.DataProcessing.service.MinIOService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.annotation.PreDestroy;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
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
public class BitcoinDataServiceImpl implements BitcoinDataService {

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

    @Autowired(required = false)
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

    private static final String CHAIN_BTC = "BTC";
    private static final int MAX_TRANSACTIONS_BATCH = 5000;

    // ============= MinIO归档相关方法 =============

    private void archiveBigQueryResult(String queryType, TableResult result) {
        if (!minioArchiveEnabled) return;
        CompletableFuture.runAsync(() -> {
            try {
                List<Map<String, Object>> dataList = new ArrayList<>();
                for (FieldValueList row : result.iterateAll()) {
                    Map<String, Object> rowData = new HashMap<>();
                    for (Field field : result.getSchema().getFields()) {
                        String fieldName = field.getName();
                        FieldValue fieldValue = row.get(fieldName);
                        if (!fieldValue.isNull()) {
                            // 关键修复：提取基本类型值
                            rowData.put(fieldName, extractFieldValue(fieldValue));
                        }
                    }
                    dataList.add(rowData);
                }
                String json = objectMapper.writeValueAsString(dataList);
                minIOService.archiveRawBigQueryResponse(queryType, json);
            } catch (Exception e) {
                log.error("归档BigQuery结果失败: {}", queryType, e);
                uploadErrorToMinio("归档BigQuery结果失败", queryType, e);
            }
        });
    }

    // 辅助方法：递归提取 FieldValue 的实际值
    private Object extractFieldValue(FieldValue fieldValue) {
        if (fieldValue.isNull()) return null;
        switch (fieldValue.getAttribute()) {
            case PRIMITIVE:
                // 基本类型：字符串、数字、布尔
                if (fieldValue.getStringValue() != null) return fieldValue.getStringValue();
                if (fieldValue.getNumericValue() != null) return fieldValue.getNumericValue();

                // 处理 bytes 字段：转为 Base64 字符串
                if (fieldValue.getBytesValue() != null) {
                    return Base64.getEncoder().encodeToString(fieldValue.getBytesValue());
                }
                return null;
            case REPEATED:
                List<Object> list = new ArrayList<>();
                for (FieldValue item : fieldValue.getRepeatedValue()) {
                    list.add(extractFieldValue(item));
                }
                return list;
            case RECORD:
                Map<String, Object> map = new LinkedHashMap<>();
                for (FieldValue item : fieldValue.getRecordValue()) {
                    // 这里需要知道字段名，但 FieldValue 不直接提供。可以依赖 schema 或跳过
                    // 简化：不深入记录内部结构，只记录字符串表示
                    map.put("value", item.getStringValue());
                }
                return map;
            default:
                return fieldValue.getStringValue();
        }
    }

    private void archiveBlockData(ChainBlock block) {
        if (!minioArchiveEnabled) return;
        CompletableFuture.runAsync(() -> {
            try {
                Map<String, Object> data = new HashMap<>();
                data.put("height", block.getHeight());
                data.put("block_hash", block.getBlockHash());
                data.put("prev_block_hash", block.getPrevBlockHash());
                data.put("block_time", block.getBlockTime());
                data.put("tx_count", block.getTxCount());
                data.put("size_bytes", block.getRawSizeBytes());
                String json = objectMapper.writeValueAsString(data);
                minIOService.archiveBlockData(block.getHeight(), json);
            } catch (Exception e) {
                log.error("归档区块数据失败: {}", block.getHeight(), e);
                uploadErrorToMinio("归档区块数据失败", "height=" + block.getHeight(), e);
            }
        });
    }

    private void archiveTransactionData(ChainTx tx) {
        if (!minioArchiveEnabled) return;
        CompletableFuture.runAsync(() -> {
            try {
                Map<String, Object> data = new HashMap<>();
                data.put("hash", tx.getTxHash());
                data.put("block_number", tx.getBlockHeight());
                data.put("block_time", tx.getBlockTime());
                data.put("total_input", tx.getTotalInput());
                data.put("total_output", tx.getTotalOutput());
                data.put("fee", tx.getFee());
                String json = objectMapper.writeValueAsString(data);
                minIOService.archiveTransactionData(tx.getTxHash(), json);
            } catch (Exception e) {
                log.error("归档交易数据失败: {}", tx.getTxHash(), e);
                uploadErrorToMinio("归档交易数据失败", "txHash=" + tx.getTxHash(), e);
            }
        });
    }

    private void archiveSyncLog(String operation, String details, boolean success) {
        if (!minioArchiveEnabled) return;
        CompletableFuture.runAsync(() -> {
            try {
                Map<String, Object> logEntry = new HashMap<>();
                logEntry.put("timestamp", LocalDateTime.now().toString());
                logEntry.put("operation", operation);
                logEntry.put("details", details);
                logEntry.put("success", success);
                logEntry.put("service", "BitcoinDataService");
                String content = objectMapper.writeValueAsString(logEntry);
                if (success) {
                    minIOService.archiveSyncLog("BTC", content);
                } else {
                    minIOService.archiveErrorLog(content);
                }
            } catch (Exception e) {
                log.error("归档同步日志失败", e);
                uploadErrorToMinio("归档同步日志失败", operation, e);
            }
        });
    }

    private void uploadErrorToMinio(String errorMessage, String context, Exception e) {
        if (!minioArchiveEnabled) return;
        CompletableFuture.runAsync(() -> {
            File tempFile = null;
            try {
                String fileName = String.format("error_%s_%d.log",
                        LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")),
                        System.nanoTime());
                tempFile = File.createTempFile("error_", ".log");
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile))) {
                    writer.write("时间: " + LocalDateTime.now().toString());
                    writer.newLine();
                    writer.write("错误: " + errorMessage);
                    writer.newLine();
                    writer.write("上下文: " + context);
                    writer.newLine();
                    writer.write("异常信息: " + e.getMessage());
                    writer.newLine();
                    writer.write("堆栈跟踪:");
                    writer.newLine();
                    for (StackTraceElement element : e.getStackTrace()) {
                        writer.write("\tat " + element.toString());
                        writer.newLine();
                    }
                }
                minIOService.uploadFile(tempFile.getAbsolutePath());
                log.info("错误日志已上传到 MinIO: {}", fileName);
            } catch (Exception ex) {
                log.error("上传错误日志到 MinIO 失败", ex);
            } finally {
                if (tempFile != null && tempFile.exists()) {
                    tempFile.delete();
                }
            }
        });
    }

    // ============= 1. 分块加载历史数据 =============

    @Override
    @Transactional
    public ApiResponse<String> syncHistoricalData(Long startHeight, Long endHeight, Integer batchDays) {
        try {
            log.info("开始同步比特币历史数据: 高度 {} 到 {}", startHeight, endHeight);
            archiveSyncLog("syncHistoricalData",
                    String.format("开始同步历史数据: %d 到 %d", startHeight, endHeight), true);

            if (startHeight == null) startHeight = 0L;
            if (endHeight == null) endHeight = 800_000L;
            if (batchDays == null || batchDays <= 0) batchDays = 1;
            if (batchDays > 30) batchDays = 30;

            LocalDate startDate = getBlockDateByHeight(startHeight);
            LocalDate endDate = getBlockDateByHeight(endHeight);
            if (startDate == null || endDate == null) {
                log.warn("无法获取区块对应日期，回退到按高度范围同步");
                return syncHistoricalDataByHeightRange(startHeight, endHeight, 1000);
            }

            int totalBlocks = 0;
            int totalTransactions = 0;

            LocalDate current = startDate;
            while (!current.isAfter(endDate)) {
                LocalDate batchEnd = current.plusDays(batchDays - 1);
                if (batchEnd.isAfter(endDate)) batchEnd = endDate;

                log.info("同步日期批次: {} 到 {}", current, batchEnd);

                int blocks = syncBlocksByDateRange(current, batchEnd);
                totalBlocks += blocks;

                for (LocalDate date = current; !date.isAfter(batchEnd); date = date.plusDays(1)) {
                    for (int startHour = 0; startHour < 24; startHour += 2) {
                        int txCount = syncTwoHourlyTransactions(date, startHour, startHour + 2);
                        totalTransactions += txCount;
                    }
                }

                current = batchEnd.plusDays(1);
                Thread.sleep(1000);
            }

            archiveSyncLog("syncHistoricalData",
                    String.format("历史数据同步完成，共同步 %d 个区块，%d 笔交易", totalBlocks, totalTransactions), true);
            return ApiResponse.success(String.format("历史数据同步完成，共同步 %d 个区块，%d 笔交易", totalBlocks, totalTransactions), null);
        } catch (Exception e) {
            log.error("同步历史数据失败", e);
            uploadErrorToMinio("同步历史数据失败",
                    String.format("startHeight=%d, endHeight=%d, batchDays=%d", startHeight, endHeight, batchDays), e);
            archiveSyncLog("syncHistoricalData", "同步失败: " + e.getMessage(), false);
            return ApiResponse.error(500, "历史数据同步失败: " + e.getMessage());
        }
    }

    private ApiResponse<String> syncHistoricalDataByHeightRange(Long startHeight, Long endHeight, int batchSize) {
        try {
            int totalBlocks = 0;
            int totalTransactions = 0;
            for (Long from = startHeight; from <= endHeight; from += batchSize) {
                Long to = Math.min(from + batchSize - 1, endHeight);
                ApiResponse<List<ChainBlock>> blocksResp = getBlocks(from, to, batchSize);
                if (blocksResp.getCode() == 200 && blocksResp.getData() != null) {
                    List<ChainBlock> blocks = blocksResp.getData();
                    totalBlocks += blocks.size();
                    for (ChainBlock block : blocks) {
                        totalTransactions += syncBlockTransactions(block.getHeight());
                    }
                }
                Thread.sleep(1000);
            }
            return ApiResponse.success(String.format("历史数据同步完成，共同步 %d 个区块，%d 笔交易", totalBlocks, totalTransactions), null);
        } catch (Exception e) {
            log.error("回退高度范围同步失败", e);
            uploadErrorToMinio("回退高度范围同步失败",
                    String.format("startHeight=%d, endHeight=%d, batchSize=%d", startHeight, endHeight, batchSize), e);
            return ApiResponse.error(500, "同步失败: " + e.getMessage());
        }
    }

    private LocalDate getBlockDateByHeight(Long height) {
        String query = String.format(
                "SELECT DATE(timestamp) as block_date FROM `bigquery-public-data.crypto_bitcoin.blocks` WHERE number = %d",
                height);
        try {
            TableResult result = executeBigQuery(query);
            for (FieldValueList row : result.iterateAll()) {
                if (!row.get("block_date").isNull()) {
                    String dateStr = row.get("block_date").getStringValue();
                    return LocalDate.parse(dateStr);
                }
            }
        } catch (Exception e) {
            log.warn("获取区块 {} 的日期失败", height, e);
            uploadErrorToMinio("获取区块日期失败", "height=" + height, e);
        }
        return null;
    }

    private int syncBlocksByDateRange(LocalDate startDate, LocalDate endDate) {
        try {
            String query = String.format(
                    "SELECT `number`, `hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                            "WHERE DATE(`timestamp`) BETWEEN '%s' AND '%s' " +
                            "ORDER BY `number` ASC",
                    startDate.toString(), endDate.toString()
            );
            TableResult result = executeBigQuery(query);
            archiveBigQueryResult("blocks_date_range", result);

            List<ChainBlock> blocks = mapToChainBlocks(result);
            int saved = 0;
            for (ChainBlock block : blocks) {
                if (!chainBlockRepository.findByChainAndHeight(CHAIN_BTC, block.getHeight()).isPresent()) {
                    block.setChain(CHAIN_BTC);
                    chainBlockRepository.save(block);
                    addToBlockBuffer(block);  // 改为批量缓冲区
                    saved++;
                }
            }
            log.info("同步区块日期范围 {} - {}，新增 {} 个区块", startDate, endDate, saved);
            return saved;
        } catch (Exception e) {
            log.error("同步区块日期范围失败: {} - {}", startDate, endDate, e);
            uploadErrorToMinio("同步区块日期范围失败", "startDate=" + startDate + ", endDate=" + endDate, e);
            return 0;
        }
    }

    private int syncTwoHourlyTransactions(LocalDate date, int startHour, int endHour) {
        LocalDateTime start = date.atTime(startHour, 0, 0);
        LocalDateTime end = start.plusHours(2);
        String startTimestamp = start.format(BIGQUERY_TIMESTAMP_FORMAT);
        String endTimestamp = end.format(BIGQUERY_TIMESTAMP_FORMAT);
        String yearMonthFirst = date.format(DateTimeFormatter.ofPattern("yyyy-MM-01"));

        String query = String.format(
                "SELECT `hash`, `block_number`, `block_timestamp`, `input_value`, `output_value`, `fee`, " +
                        "`lock_time`, `size`, `inputs`, `outputs` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.transactions` " +
                        "WHERE `block_timestamp` >= TIMESTAMP('%s') AND `block_timestamp` < TIMESTAMP('%s') " +
                        "  AND `block_timestamp_month` = '%s' " +
                        "ORDER BY `block_number`, `block_timestamp`",
                startTimestamp, endTimestamp, yearMonthFirst
        );

        try {
            TableResult result = executeBigQuery(query);
            archiveBigQueryResult("two_hourly_transactions", result);

            Set<String> existingTxHashes = chainTxRepository.findTxHashesByTimeRange(CHAIN_BTC, start, end);
            List<ChainTx> batchTxs = new ArrayList<>(MAX_TRANSACTIONS_BATCH);
            Map<String, List<ChainTxInput>> batchInputsMap = new HashMap<>();
            Map<String, List<ChainTxOutput>> batchOutputsMap = new HashMap<>();

            for (FieldValueList row : result.iterateAll()) {
                ChainTx tx = mapToChainTx(row);
                if (existingTxHashes.contains(tx.getTxHash())) {
                    continue;
                }
                List<ChainTxInput> inputs = parseInputs(row, tx);
                List<ChainTxOutput> outputs = parseOutputs(row, tx);

                batchTxs.add(tx);
                batchInputsMap.put(tx.getTxHash(), inputs);
                batchOutputsMap.put(tx.getTxHash(), outputs);

                if (batchTxs.size() >= MAX_TRANSACTIONS_BATCH) {
                    saveTransactionsBatch(batchTxs, batchInputsMap, batchOutputsMap);
                    batchTxs.clear();
                    batchInputsMap.clear();
                    batchOutputsMap.clear();
                }
            }

            if (!batchTxs.isEmpty()) {
                saveTransactionsBatch(batchTxs, batchInputsMap, batchOutputsMap);
            }

            log.info("日期 {} 时段 {}-{} 同步比特币交易 {} 笔", date, startHour, endHour, batchTxs.size());
            return batchTxs.size();
        } catch (Exception e) {
            log.error("同步时段比特币交易失败: {} {}-{}", date, startHour, endHour, e);
            uploadErrorToMinio("同步时段比特币交易失败",
                    String.format("date=%s, startHour=%d, endHour=%d", date, startHour, endHour), e);
            return 0;
        }
    }

    private void saveTransactionsBatch(List<ChainTx> txs,
                                       Map<String, List<ChainTxInput>> inputsMap,
                                       Map<String, List<ChainTxOutput>> outputsMap) {
        if (txs.isEmpty()) return;
        try {
            for (ChainTx tx : txs) {
                tx.setChain(CHAIN_BTC);
            }
            chainTxRepository.saveAll(txs);
            log.info("批量保存 {} 笔比特币交易到 MySQL", txs.size());

            List<ChainTxInput> allInputs = new ArrayList<>();
            List<ChainTxOutput> allOutputs = new ArrayList<>();
            for (ChainTx tx : txs) {
                List<ChainTxInput> inputs = inputsMap.get(tx.getTxHash());
                if (inputs != null) {
                    for (ChainTxInput input : inputs) {
                        input.setTransaction(tx);
                        input.setChain(CHAIN_BTC);
                        allInputs.add(input);
                    }
                }
                List<ChainTxOutput> outputs = outputsMap.get(tx.getTxHash());
                if (outputs != null) {
                    for (ChainTxOutput output : outputs) {
                        output.setTransaction(tx);
                        output.setChain(CHAIN_BTC);
                        allOutputs.add(output);
                    }
                }
                addToTxBuffer(tx);  // 改为批量缓冲区
            }
            if (!allInputs.isEmpty()) chainTxInputRepository.saveAll(allInputs);
            if (!allOutputs.isEmpty()) chainTxOutputRepository.saveAll(allOutputs);

            if (graphService != null) {
                try {
                    log.info("开始保存 {} 笔比特币交易到图数据库", txs.size());
                    long start = System.currentTimeMillis();
                    graphService.saveBitcoinTransactionsToGraph(txs, inputsMap, outputsMap);
                    log.info("保存 {} 笔交易到图数据库完成，耗时 {} ms", txs.size(), System.currentTimeMillis() - start);
                } catch (Exception e) {
                    log.error("批量保存比特币交易到图数据库失败", e);
                    uploadErrorToMinio("批量保存比特币交易到图数据库失败", "batchSize=" + txs.size(), e);
                }
            }
        } catch (Exception e) {
            log.error("批量保存比特币交易失败", e);
            uploadErrorToMinio("批量保存比特币交易失败", "batchSize=" + txs.size(), e);
        }
    }

    // ============= 2. 获取最新数据 =============

    @Override
    @Transactional
    public ApiResponse<Map<String, Object>> getLatestData() {
        try {
            Map<String, Object> result = new HashMap<>();

            Long dbLatestHeight = chainBlockRepository.findMaxHeight(CHAIN_BTC);
            LocalDateTime dbLatestTime = chainBlockRepository.findLatestBlockTime(CHAIN_BTC);

            result.put("db_latest_height", dbLatestHeight);
            result.put("db_latest_time", dbLatestTime);

            String latestBlockQuery =
                    "SELECT MAX(`number`) as max_height, MAX(`timestamp`) as max_time " +
                            "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                            "WHERE DATE(`timestamp`) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)";
            TableResult bqResult = executeBigQuery(latestBlockQuery);
            archiveBigQueryResult("latest_block_query", bqResult);

            for (FieldValueList row : bqResult.iterateAll()) {
                if (!row.get("max_height").isNull()) {
                    result.put("bq_latest_height", row.get("max_height").getLongValue());
                }
                if (!row.get("max_time").isNull()) {
                    result.put("bq_latest_time", parseBigQueryTimestamp(row.get("max_time").getStringValue()));
                }
            }

            long dbHeight = dbLatestHeight != null ? dbLatestHeight : 0L;
            long bqHeight = (Long) result.getOrDefault("bq_latest_height", 0L);
            result.put("behind_blocks", bqHeight - dbHeight);

            if (dbLatestTime != null && result.get("bq_latest_time") != null) {
                LocalDateTime bqTime = (LocalDateTime) result.get("bq_latest_time");
                long hours = java.time.Duration.between(dbLatestTime, bqTime).toHours();
                result.put("behind_hours", hours);
            }

            return ApiResponse.success(result, null);
        } catch (Exception e) {
            log.error("获取最新数据失败", e);
            uploadErrorToMinio("获取最新数据失败", "", e);
            return ApiResponse.error(500, "获取最新数据失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public ApiResponse<String> syncLatestData(Integer blocksToSync) {
        try {
            if (blocksToSync == null || blocksToSync <= 0) blocksToSync = 100;
            if (blocksToSync > 1000) blocksToSync = 1000;

            Long latestHeightInDB = chainBlockRepository.findMaxHeight(CHAIN_BTC);
            if (latestHeightInDB == null) latestHeightInDB = 0L;

            log.info("开始同步最新数据，数据库最新高度: {}，同步 {} 个区块", latestHeightInDB, blocksToSync);
            archiveSyncLog("syncLatestData",
                    String.format("开始同步最新数据，数据库最新高度: %d，同步 %d 个区块", latestHeightInDB, blocksToSync), true);

            String blocksQuery = String.format(
                    "SELECT `number`, `hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                            "WHERE `number` > %d " +
                            "  AND DATE(`timestamp`) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) " +
                            "ORDER BY `number` ASC " +
                            "LIMIT %d",
                    latestHeightInDB, blocksToSync
            );

            TableResult blocksResult = executeBigQuery(blocksQuery);
            archiveBigQueryResult("latest_blocks_query", blocksResult);

            List<ChainBlock> blocks = mapToChainBlocks(blocksResult);
            if (blocks.isEmpty()) {
                return ApiResponse.success("已是最新数据，无需同步", null);
            }

            int savedBlocks = 0;
            int savedTransactions = 0;

            for (ChainBlock block : blocks) {
                if (!chainBlockRepository.findByChainAndHeight(CHAIN_BTC, block.getHeight()).isPresent()) {
                    block.setChain(CHAIN_BTC);
                    chainBlockRepository.save(block);
                    addToBlockBuffer(block);  // 改为批量缓冲区
                    savedBlocks++;
                    savedTransactions += syncBlockTransactions(block.getHeight());
                }
            }

            archiveSyncLog("syncLatestData",
                    String.format("同步完成，新增 %d 个区块，%d 笔交易", savedBlocks, savedTransactions), true);
            return ApiResponse.success(String.format("同步完成，新增 %d 个区块，%d 笔交易", savedBlocks, savedTransactions), null);
        } catch (Exception e) {
            log.error("同步最新数据失败", e);
            uploadErrorToMinio("同步最新数据失败", "blocksToSync=" + blocksToSync, e);
            archiveSyncLog("syncLatestData", "同步失败: " + e.getMessage(), false);
            return ApiResponse.error(500, "同步最新数据失败: " + e.getMessage());
        }
    }

    private int syncBlockTransactions(Long blockHeight) {
        try {
            Optional<ChainBlock> blockOpt = chainBlockRepository.findByChainAndHeight(CHAIN_BTC, blockHeight);
            if (!blockOpt.isPresent()) {
                log.warn("区块 {} 不在数据库中，跳过交易同步", blockHeight);
                return 0;
            }
            LocalDateTime blockTime = blockOpt.get().getBlockTime();
            LocalDate blockDate = blockTime.toLocalDate();
            String startDate = blockDate.toString();
            String endDate = blockDate.plusDays(1).toString();
            String yearMonthFirst = blockTime.format(DateTimeFormatter.ofPattern("yyyy-MM-01"));

            String query = String.format(
                    "SELECT `hash`, `block_number`, `block_timestamp`, `input_value`, `output_value`, `fee`, " +
                            "`lock_time`, `size`, `inputs`, `outputs` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.transactions` " +
                            "WHERE `block_number` = %d " +
                            "  AND `block_timestamp_month` = '%s' " +
                            "  AND `block_timestamp` >= '%s' AND `block_timestamp` < '%s' " +
                            "ORDER BY `block_timestamp`",
                    blockHeight, yearMonthFirst, startDate, endDate
            );
            TableResult result = executeBigQuery(query);
            archiveBigQueryResult("block_transactions_query", result);

            List<ChainTx> transactions = new ArrayList<>();
            List<ChainTx> batchForGraph = new ArrayList<>();
            Map<String, List<ChainTxInput>> inputsMap = new HashMap<>();
            Map<String, List<ChainTxOutput>> outputsMap = new HashMap<>();

            for (FieldValueList row : result.iterateAll()) {
                ChainTx tx = mapToChainTx(row);
                List<ChainTxInput> inputs = parseInputs(row, tx);
                List<ChainTxOutput> outputs = parseOutputs(row, tx);
                transactions.add(tx);

                if (!chainTxRepository.findByChainAndTxHash(CHAIN_BTC, tx.getTxHash()).isPresent()) {
                    tx.setChain(CHAIN_BTC);
                    ChainTx savedTx = chainTxRepository.save(tx);
                    addToTxBuffer(savedTx);  // 改为批量缓冲区

                    for (ChainTxInput input : inputs) {
                        input.setTransaction(savedTx);
                        input.setChain(CHAIN_BTC);
                        chainTxInputRepository.save(input);
                    }
                    for (ChainTxOutput output : outputs) {
                        output.setTransaction(savedTx);
                        output.setChain(CHAIN_BTC);
                        chainTxOutputRepository.save(output);
                    }

                    batchForGraph.add(savedTx);
                    inputsMap.put(savedTx.getTxHash(), inputs);
                    outputsMap.put(savedTx.getTxHash(), outputs);
                }
            }

            if (!batchForGraph.isEmpty() && graphService != null) {
                try {
                    graphService.saveBitcoinTransactionsToGraph(batchForGraph, inputsMap, outputsMap);
                } catch (Exception e) {
                    log.error("批量保存比特币交易到图数据库失败", e);
                    uploadErrorToMinio("批量保存比特币交易到图数据库失败", "blockHeight=" + blockHeight, e);
                }
            }

            log.debug("同步区块 {} 的 {} 笔交易", blockHeight, transactions.size());
            return transactions.size();
        } catch (Exception e) {
            log.error("同步区块交易失败: {}", blockHeight, e);
            uploadErrorToMinio("同步区块交易失败", "blockHeight=" + blockHeight, e);
            return 0;
        }
    }

    // ============= 3. 定时同步 =============

//    @Scheduled(cron = "0 0 4 * * ?")
    @Transactional
    public void scheduledSyncLatestData() {
        try {
            log.info("开始定时同步最新数据");
            archiveSyncLog("scheduledSyncLatestData", "开始定时同步最新数据", true);

            Long latestHeightInDB = chainBlockRepository.findMaxHeight(CHAIN_BTC);
            if (latestHeightInDB == null) {
                log.info("数据库无数据，跳过定时同步");
                return;
            }

            String blocksQuery = String.format(
                    "SELECT `number`, `hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                            "WHERE `number` > %d " +
                            "  AND DATE(`timestamp`) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) " +
                            "ORDER BY `number` ASC " +
                            "LIMIT 20",
                    latestHeightInDB
            );
            TableResult blocksResult = executeBigQuery(blocksQuery);
            archiveBigQueryResult("scheduled_blocks_query", blocksResult);

            List<ChainBlock> blocks = mapToChainBlocks(blocksResult);
            if (!blocks.isEmpty()) {
                int savedBlocks = 0;
                for (ChainBlock block : blocks) {
                    if (!chainBlockRepository.findByChainAndHeight(CHAIN_BTC, block.getHeight()).isPresent()) {
                        block.setChain(CHAIN_BTC);
                        chainBlockRepository.save(block);
                        addToBlockBuffer(block);  // 改为批量缓冲区
                        syncBlockTransactions(block.getHeight());
                        savedBlocks++;
                    }
                }
                log.info("定时同步完成，新增 {} 个区块", savedBlocks);
                archiveSyncLog("scheduledSyncLatestData", String.format("定时同步完成，新增 %d 个区块", savedBlocks), true);
            } else {
                log.info("定时同步：无新数据");
            }
        } catch (Exception e) {
            log.error("定时同步失败", e);
            uploadErrorToMinio("定时同步失败", "", e);
            archiveSyncLog("scheduledSyncLatestData", "定时同步失败: " + e.getMessage(), false);
        }
    }

    // ============= 4. 数据库查询功能 =============

    @Override
    @Cacheable(value = "bitcoinStats", key = "'stats'")
    public ApiResponse<Map<String, Object>> getBlockchainStats() {
        try {
            Map<String, Object> stats = new HashMap<>();
            Long blockCount = chainBlockRepository.countByChain(CHAIN_BTC);
            Long txCount = chainTxRepository.countByChain(CHAIN_BTC);
            Long latestHeight = chainBlockRepository.findMaxHeight(CHAIN_BTC);
            LocalDateTime latestTime = chainBlockRepository.findLatestBlockTime(CHAIN_BTC);

            stats.put("chain", "Bitcoin");
            stats.put("blockCount", blockCount != null ? blockCount : 0);
            stats.put("transactionCount", txCount != null ? txCount : 0);
            stats.put("latestBlockHeight", latestHeight != null ? latestHeight : 0);
            stats.put("latestBlockTime", latestTime);
            stats.put("timestamp", LocalDateTime.now());
            return ApiResponse.success(stats, null);
        } catch (Exception e) {
            log.error("获取区块链统计失败", e);
            uploadErrorToMinio("获取区块链统计失败", "", e);
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
            Page<ChainBlock> blocks = chainBlockRepository.findByChain(CHAIN_BTC, pageable);
            return ApiResponse.success(blocks, blocks.getTotalElements());
        } catch (Exception e) {
            log.error("分页查询区块失败", e);
            uploadErrorToMinio("分页查询区块失败", "page=" + page + ", size=" + size, e);
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
            Page<ChainTx> transactions = chainTxRepository.findByChain(CHAIN_BTC, pageable);
            return ApiResponse.success(transactions, transactions.getTotalElements());
        } catch (Exception e) {
            log.error("分页查询交易失败", e);
            uploadErrorToMinio("分页查询交易失败", "page=" + page + ", size=" + size, e);
            return ApiResponse.error(500, "分页查询交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainTx>> getTransactionsByAddress(String address, Integer limit) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;
            Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "blockHeight"));
            Page<ChainTx> txPage = chainTxRepository.findByFromAddressOrToAddress(CHAIN_BTC, address, pageable);
            List<ChainTx> txs = txPage.getContent();
            return ApiResponse.success(txs, txPage.getTotalElements());
        } catch (Exception e) {
            log.error("按地址查询交易失败", e);
            uploadErrorToMinio("按地址查询交易失败", "address=" + address, e);
            return ApiResponse.error(500, "按地址查询交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressBalance(String address) {
        try {
            BigDecimal totalInput = chainTxInputRepository.sumValueByAddress(address);
            BigDecimal totalOutput = chainTxOutputRepository.sumValueByAddress(address);
            BigDecimal balance = (totalOutput != null ? totalOutput : BigDecimal.ZERO)
                    .subtract(totalInput != null ? totalInput : BigDecimal.ZERO);
            Map<String, Object> result = new HashMap<>();
            result.put("address", address);
            result.put("totalReceived", totalOutput != null ? totalOutput : BigDecimal.ZERO);
            result.put("totalSent", totalInput != null ? totalInput : BigDecimal.ZERO);
            result.put("balance", balance);
            result.put("transactionCount", chainTxRepository.countByFromAddressOrToAddress(CHAIN_BTC, address));
            return ApiResponse.success(result, null);
        } catch (Exception e) {
            log.error("查询地址余额失败", e);
            uploadErrorToMinio("查询地址余额失败", "address=" + address, e);
            return ApiResponse.error(500, "查询地址余额失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressInfo(String address) {
        try {
            Map<String, Object> info = new HashMap<>();
            info.put("address", address);
            info.put("balance", getAddressBalance(address).getData());
            Pageable pageable = PageRequest.of(0, 10, Sort.by(Sort.Direction.DESC, "blockTime"));
            Page<ChainTx> recentTxs = chainTxRepository.findByFromAddressOrToAddress(CHAIN_BTC, address, pageable);
            info.put("recentTransactions", recentTxs.getContent());
            info.put("transactionCount", recentTxs.getTotalElements());
            return ApiResponse.success(info, null);
        } catch (Exception e) {
            log.error("查询地址信息失败", e);
            uploadErrorToMinio("查询地址信息失败", "address=" + address, e);
            return ApiResponse.error(500, "查询地址信息失败: " + e.getMessage());
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

            LocalDate current = startDate;
            while (!current.isAfter(endDate)) {
                LocalDateTime dayStart = current.atStartOfDay();
                LocalDateTime dayEnd = current.plusDays(1).atStartOfDay();

                Long blocks = chainBlockRepository.countByChainAndBlockTimeBetween(CHAIN_BTC, dayStart, dayEnd);
                Long transactions = chainTxRepository.countByChainAndBlockTimeBetween(CHAIN_BTC, dayStart, dayEnd);

                Map<String, Object> dayStat = new HashMap<>();
                dayStat.put("date", current.toString());
                dayStat.put("blocks", blocks != null ? blocks : 0);
                dayStat.put("transactions", transactions != null ? transactions : 0);
                dailyStats.add(dayStat);
                current = current.plusDays(1);
            }
            return ApiResponse.success(dailyStats, (long) dailyStats.size());
        } catch (Exception e) {
            log.error("获取每日统计失败", e);
            uploadErrorToMinio("获取每日统计失败", "days=" + days, e);
            return ApiResponse.error(500, "获取每日统计失败: " + e.getMessage());
        }
    }

    // ============= 基础查询方法（数据库优先） =============

    @Override
    public ApiResponse<List<ChainBlock>> getBlocks(Long startHeight, Long endHeight, Integer limit) {
        try {
            if (startHeight == null) startHeight = 0L;
            if (endHeight == null) endHeight = Long.MAX_VALUE;
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;

            Pageable pageable = PageRequest.of(0, limit, Sort.by("height").descending());
            List<ChainBlock> dbBlocks = chainBlockRepository.findByChainAndHeightBetween(CHAIN_BTC, startHeight, endHeight, pageable);
            if (!dbBlocks.isEmpty()) {
                return ApiResponse.success(dbBlocks, (long) dbBlocks.size());
            }

            String query = String.format(
                    "SELECT `number`, `hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                            "WHERE `number` >= %d AND `number` <= %d " +
                            "ORDER BY `number` DESC " +
                            "LIMIT %d",
                    startHeight, endHeight, limit
            );
            TableResult result = executeBigQuery(query);
            archiveBigQueryResult("get_blocks_query", result);
            List<ChainBlock> blocks = mapToChainBlocks(result);

            for (ChainBlock block : blocks) {
                if (!chainBlockRepository.findByChainAndHeight(CHAIN_BTC, block.getHeight()).isPresent()) {
                    block.setChain(CHAIN_BTC);
                    chainBlockRepository.save(block);
                    addToBlockBuffer(block);  // 改为批量缓冲区
                }
            }
            return ApiResponse.success(blocks, (long) blocks.size());
        } catch (Exception e) {
            log.error("获取区块数据失败", e);
            uploadErrorToMinio("获取区块数据失败",
                    String.format("startHeight=%d, endHeight=%d, limit=%d", startHeight, endHeight, limit), e);
            return ApiResponse.error(500, "获取区块数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainTx>> getTransactions(Long blockHeight, Integer limit, Integer offset) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;
            if (offset == null || offset < 0) offset = 0;

            Pageable pageable = PageRequest.of(offset / limit, limit, Sort.by("txIndex").ascending());
            List<ChainTx> dbTxs = chainTxRepository.findByChainAndBlockHeight(CHAIN_BTC, blockHeight, pageable);
            if (!dbTxs.isEmpty()) {
                return ApiResponse.success(dbTxs, (long) dbTxs.size());
            }

            Optional<ChainBlock> blockOpt = chainBlockRepository.findByChainAndHeight(CHAIN_BTC, blockHeight);
            if (!blockOpt.isPresent()) {
                log.warn("区块 {} 不在数据库中，无法从BigQuery获取交易", blockHeight);
                return ApiResponse.success(Collections.emptyList(), 0L);
            }
            LocalDateTime blockTime = blockOpt.get().getBlockTime();
            LocalDate blockDate = blockTime.toLocalDate();
            String startDate = blockDate.toString();
            String endDate = blockDate.plusDays(1).toString();
            String yearMonthFirst = blockTime.format(DateTimeFormatter.ofPattern("yyyy-MM-01"));

            String query = String.format(
                    "SELECT `hash`, `block_number`, `block_timestamp`, `input_value`, `output_value`, `fee`, " +
                            "`lock_time`, `size`, `inputs`, `outputs` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.transactions` " +
                            "WHERE `block_number` = %d " +
                            "  AND `block_timestamp_month` = '%s' " +
                            "  AND `block_timestamp` >= '%s' AND `block_timestamp` < '%s' " +
                            "ORDER BY `block_timestamp` " +
                            "LIMIT %d OFFSET %d",
                    blockHeight, yearMonthFirst, startDate, endDate, limit, offset
            );
            TableResult result = executeBigQuery(query);
            archiveBigQueryResult("get_transactions_query", result);

            List<ChainTx> txs = new ArrayList<>();
            for (FieldValueList row : result.iterateAll()) {
                ChainTx tx = mapToChainTx(row);
                List<ChainTxInput> inputs = parseInputs(row, tx);
                List<ChainTxOutput> outputs = parseOutputs(row, tx);
                txs.add(tx);

                if (!chainTxRepository.findByChainAndTxHash(CHAIN_BTC, tx.getTxHash()).isPresent()) {
                    tx.setChain(CHAIN_BTC);
                    ChainTx savedTx = chainTxRepository.save(tx);
                    addToTxBuffer(savedTx);  // 改为批量缓冲区

                    for (ChainTxInput input : inputs) {
                        input.setTransaction(savedTx);
                        input.setChain(CHAIN_BTC);
                        chainTxInputRepository.save(input);
                    }
                    for (ChainTxOutput output : outputs) {
                        output.setTransaction(savedTx);
                        output.setChain(CHAIN_BTC);
                        chainTxOutputRepository.save(output);
                    }
                }
            }
            return ApiResponse.success(txs, (long) txs.size());
        } catch (Exception e) {
            log.error("获取交易数据失败", e);
            uploadErrorToMinio("获取交易数据失败",
                    String.format("blockHeight=%d, limit=%d, offset=%d", blockHeight, limit, offset), e);
            return ApiResponse.error(500, "获取交易数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainBlock>> getBlocksByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;
            Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "height"));
            List<ChainBlock> blocks = chainBlockRepository.findByChainAndBlockTimeBetween(CHAIN_BTC, startTime, endTime, pageable);
            return ApiResponse.success(blocks, (long) blocks.size());
        } catch (Exception e) {
            log.error("按时间获取区块失败", e);
            uploadErrorToMinio("按时间获取区块失败",
                    String.format("startTime=%s, endTime=%s, limit=%d", startTime, endTime, limit), e);
            return ApiResponse.error(500, "按时间获取区块失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainTx>> getTransactionsByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;
            Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "blockHeight", "txIndex"));
            Page<ChainTx> txPage = chainTxRepository.findByChainAndBlockTimeBetween(CHAIN_BTC, startTime, endTime, pageable);
            List<ChainTx> txs = txPage.getContent();
            return ApiResponse.success(txs, (long) txs.size());
        } catch (Exception e) {
            log.error("按时间获取交易失败", e);
            uploadErrorToMinio("按时间获取交易失败",
                    String.format("startTime=%s, endTime=%s, limit=%d", startTime, endTime, limit), e);
            return ApiResponse.error(500, "按时间获取交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getTransactionDetail(String txHash) {
        try {
            Optional<ChainTx> txOpt = chainTxRepository.findByChainAndTxHash(CHAIN_BTC, txHash);
            if (!txOpt.isPresent()) {
                return ApiResponse.error(404, "交易不存在: " + txHash);
            }
            ChainTx tx = txOpt.get();
            List<ChainTxInput> inputs = chainTxInputRepository.findByTransaction(tx);
            List<ChainTxOutput> outputs = chainTxOutputRepository.findByTransaction(tx);

            Map<String, Object> detail = new HashMap<>();
            detail.put("transaction", tx);
            detail.put("inputs", inputs);
            detail.put("outputs", outputs);
            detail.put("inputCount", inputs.size());
            detail.put("outputCount", outputs.size());
            detail.put("totalInput", tx.getTotalInput());
            detail.put("totalOutput", tx.getTotalOutput());

            return ApiResponse.success(detail, null);
        } catch (Exception e) {
            log.error("获取交易详情失败", e);
            uploadErrorToMinio("获取交易详情失败", "txHash=" + txHash, e);
            return ApiResponse.error(500, "获取交易详情失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Long> getBlockNumber() {
        try {
            Long latest = chainBlockRepository.findMaxHeight(CHAIN_BTC);
            return ApiResponse.success(latest, null);
        } catch (Exception e) {
            log.error("获取区块高度失败", e);
            uploadErrorToMinio("获取区块高度失败", "", e);
            return ApiResponse.error(500, "获取区块高度失败");
        }
    }

    @Override
    public ApiResponse<String> exportBlocksToCsv(Long startHeight, Long endHeight) {
        return ApiResponse.success("从数据库导出功能待实现", null);
    }

    @Override
    public ApiResponse<String> exportTransactionsToCsv(LocalDateTime startTime, LocalDateTime endTime) {
        return ApiResponse.success("从数据库导出功能待实现", null);
    }

    // ============= 数据映射与工具方法 =============

    private List<ChainBlock> mapToChainBlocks(TableResult result) {
        List<ChainBlock> blocks = new ArrayList<>();
        for (FieldValueList row : result.iterateAll()) {
            try {
                ChainBlock block = new ChainBlock();
                block.setHeight(row.get("number").getLongValue());
                block.setBlockHash(row.get("hash").getStringValue());
                String ts = row.get("timestamp").getStringValue();
                block.setBlockTime(parseBigQueryTimestamp(ts));
                block.setTxCount((int) row.get("transaction_count").getLongValue());
                block.setRawSizeBytes(row.get("size").getLongValue());
                blocks.add(block);
            } catch (Exception e) {
                log.error("映射区块数据失败", e);
                uploadErrorToMinio("映射区块数据失败", "", e);
            }
        }
        return blocks;
    }

    private ChainTx mapToChainTx(FieldValueList row) {
        ChainTx tx = new ChainTx();
        try {
            tx.setTxHash(row.get("hash").getStringValue());
            tx.setBlockHeight(row.get("block_number").getLongValue());
            String ts = row.get("block_timestamp").getStringValue();
            tx.setBlockTime(parseBigQueryTimestamp(ts));

            FieldValue inputValueField = row.get("input_value");
            FieldValue outputValueField = row.get("output_value");
            FieldValue feeField = row.get("fee");

            BigDecimal inputValue = inputValueField.isNull() ? BigDecimal.ZERO : BigDecimal.valueOf(inputValueField.getDoubleValue());
            BigDecimal outputValue = outputValueField.isNull() ? BigDecimal.ZERO : BigDecimal.valueOf(outputValueField.getDoubleValue());
            BigDecimal fee = feeField.isNull() ? BigDecimal.ZERO : BigDecimal.valueOf(feeField.getDoubleValue());

            tx.setTotalInput(convertSatoshiToBtc(inputValue));
            tx.setTotalOutput(convertSatoshiToBtc(outputValue));
            tx.setFee(convertSatoshiToBtc(fee));

            if (!row.get("lock_time").isNull()) {
                tx.setLocktime(row.get("lock_time").getLongValue());
            }
            if (!row.get("size").isNull()) {
                tx.setSizeBytes(row.get("size").getLongValue());
            }
        } catch (Exception e) {
            log.error("映射交易数据失败", e);
            uploadErrorToMinio("映射交易数据失败", "", e);
        }
        return tx;
    }

    private List<ChainTxInput> parseInputs(FieldValueList row, ChainTx tx) {
        List<ChainTxInput> inputs = new ArrayList<>();
        try {
            FieldValue inputsField = row.get("inputs");
            if (inputsField.isNull()) return inputs;

            for (FieldValue inputRecord : inputsField.getRepeatedValue()) {
                List<FieldValue> fields = inputRecord.getRecordValue();
                ChainTxInput input = new ChainTxInput();
                input.setChain(CHAIN_BTC);
                input.setTransaction(tx);

                if (fields.size() > 0 && !fields.get(0).isNull()) {
                    input.setInputIndex((int) fields.get(0).getLongValue());
                }
                if (fields.size() > 1 && !fields.get(1).isNull()) {
                    input.setPrevTxHash(fields.get(1).getStringValue());
                }
                if (fields.size() > 2 && !fields.get(2).isNull()) {
                    input.setPrevOutIndex((int) fields.get(2).getLongValue());
                }
                if (fields.size() > 4 && !fields.get(4).isNull()) {
                    input.setScriptSig(fields.get(4).getStringValue());
                } else if (fields.size() > 3 && !fields.get(3).isNull()) {
                    input.setScriptSig(fields.get(3).getStringValue());
                }
                if (fields.size() > 8 && !fields.get(8).isNull()) {
                    List<FieldValue> addresses = fields.get(8).getRepeatedValue();
                    if (!addresses.isEmpty()) {
                        input.setAddress(addresses.get(0).getStringValue());
                    }
                }
                if (fields.size() > 9 && !fields.get(9).isNull()) {
                    BigDecimal satoshi = BigDecimal.valueOf(fields.get(9).getDoubleValue());
                    input.setValue(convertSatoshiToBtc(satoshi));
                }
                inputs.add(input);
            }
        } catch (Exception e) {
            log.error("解析inputs失败", e);
            uploadErrorToMinio("解析inputs失败", "txHash=" + tx.getTxHash(), e);
        }
        return inputs;
    }

    private List<ChainTxOutput> parseOutputs(FieldValueList row, ChainTx tx) {
        List<ChainTxOutput> outputs = new ArrayList<>();
        try {
            FieldValue outputsField = row.get("outputs");
            if (outputsField.isNull()) return outputs;

            for (FieldValue outputRecord : outputsField.getRepeatedValue()) {
                List<FieldValue> fields = outputRecord.getRecordValue();
                ChainTxOutput output = new ChainTxOutput();
                output.setChain(CHAIN_BTC);
                output.setTransaction(tx);

                if (fields.size() > 0 && !fields.get(0).isNull()) {
                    output.setOutputIndex((int) fields.get(0).getLongValue());
                }
                if (fields.size() > 2 && !fields.get(2).isNull()) {
                    output.setScriptPubKey(fields.get(2).getStringValue());
                } else if (fields.size() > 1 && !fields.get(1).isNull()) {
                    output.setScriptPubKey(fields.get(1).getStringValue());
                }
                if (fields.size() > 5 && !fields.get(5).isNull()) {
                    List<FieldValue> addresses = fields.get(5).getRepeatedValue();
                    if (!addresses.isEmpty()) {
                        output.setAddress(addresses.get(0).getStringValue());
                    }
                }
                if (fields.size() > 6 && !fields.get(6).isNull()) {
                    BigDecimal satoshi = BigDecimal.valueOf(fields.get(6).getDoubleValue());
                    output.setValue(convertSatoshiToBtc(satoshi));
                }
                outputs.add(output);
            }
        } catch (Exception e) {
            log.error("解析outputs失败", e);
            uploadErrorToMinio("解析outputs失败", "txHash=" + tx.getTxHash(), e);
        }
        return outputs;
    }

    private BigDecimal convertSatoshiToBtc(BigDecimal satoshi) {
        return satoshi.divide(BigDecimal.valueOf(100_000_000), 8, RoundingMode.HALF_UP);
    }

    private TableResult executeBigQuery(String query) throws InterruptedException {
        try {
            QueryJobConfiguration config = QueryJobConfiguration.newBuilder(query).build();
            return bigQuery.query(config);
        } catch (Exception e) {
            log.error("BigQuery查询失败 - SQL: {}", query);
            uploadErrorToMinio("BigQuery查询失败", query, e);
            throw e;
        }
    }

    private LocalDateTime parseBigQueryTimestamp(String timestampStr) {
        try {
            String cleaned = timestampStr.trim().replace(" UTC", "");
            if (cleaned.matches("-?\\d+(\\.\\d+)?([Ee][+-]?\\d+)?")) {
                double seconds = Double.parseDouble(cleaned);
                long millis = (long) (seconds * 1000);
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(millis), ZoneId.of("UTC"));
            } else {
                try {
                    DateTimeFormatter formatterWithMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
                    return LocalDateTime.parse(cleaned, formatterWithMillis);
                } catch (Exception e1) {
                    try {
                        DateTimeFormatter formatterWithoutMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
                        return LocalDateTime.parse(cleaned, formatterWithoutMillis);
                    } catch (Exception e2) {
                        DateTimeFormatter isoFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
                        return LocalDateTime.parse(cleaned, isoFormatter);
                    }
                }
            }
        } catch (Exception e) {
            log.warn("解析时间戳失败: {}, 使用当前时间", timestampStr, e);
            uploadErrorToMinio("解析时间戳失败", "timestampStr=" + timestampStr, e);
            return LocalDateTime.now();
        }
    }

    // 批量缓冲区
    private final List<ChainBlock> blockBuffer = Collections.synchronizedList(new ArrayList<>());
    private final List<ChainTx> txBuffer = Collections.synchronizedList(new ArrayList<>());
    private static final int BLOCK_BATCH_SIZE = 100;
    private static final int TX_BATCH_SIZE = 5000;

    private void addToBlockBuffer(ChainBlock block) {
        synchronized (blockBuffer) {
            blockBuffer.add(block);
            if (blockBuffer.size() >= BLOCK_BATCH_SIZE) {
                flushBlockBuffer();
            }
        }
    }

    private void flushBlockBuffer() {
        List<ChainBlock> toUpload;
        synchronized (blockBuffer) {
            if (blockBuffer.isEmpty()) return;
            toUpload = new ArrayList<>(blockBuffer);
            blockBuffer.clear();
        }
        CompletableFuture.runAsync(() -> {
            try {
                List<Map<String, Object>> dataList = new ArrayList<>();
                for (ChainBlock block : toUpload) {
                    Map<String, Object> data = new HashMap<>();
                    data.put("height", block.getHeight());
                    data.put("block_hash", block.getBlockHash());
                    data.put("prev_block_hash", block.getPrevBlockHash());
                    data.put("block_time", block.getBlockTime());
                    data.put("tx_count", block.getTxCount());
                    data.put("size_bytes", block.getRawSizeBytes());
                    dataList.add(data);
                }
                long minHeight = toUpload.stream().mapToLong(ChainBlock::getHeight).min().orElse(0);
                long maxHeight = toUpload.stream().mapToLong(ChainBlock::getHeight).max().orElse(0);
                String fileName = String.format("blocks_%d_%d.json", minHeight, maxHeight);
                String jsonArray = objectMapper.writeValueAsString(dataList);
                minIOService.archiveBlockBatch(fileName, jsonArray);
                log.info("批量上传 {} 个区块到 MinIO: {}", toUpload.size(), fileName);
            } catch (Exception e) {
                log.error("批量上传区块失败", e);
            }
        });
    }

    private void addToTxBuffer(ChainTx tx) {
        synchronized (txBuffer) {
            txBuffer.add(tx);
            if (txBuffer.size() >= TX_BATCH_SIZE) {
                flushTxBuffer();
            }
        }
    }

    private void flushTxBuffer() {
        List<ChainTx> toUpload;
        synchronized (txBuffer) {
            if (txBuffer.isEmpty()) return;
            toUpload = new ArrayList<>(txBuffer);
            txBuffer.clear();
        }
        CompletableFuture.runAsync(() -> {
            try {
                List<Map<String, Object>> dataList = new ArrayList<>();
                for (ChainTx tx : toUpload) {
                    Map<String, Object> data = new HashMap<>();
                    data.put("hash", tx.getTxHash());
                    data.put("block_number", tx.getBlockHeight());
                    data.put("block_time", tx.getBlockTime());
                    data.put("total_input", tx.getTotalInput());
                    data.put("total_output", tx.getTotalOutput());
                    data.put("fee", tx.getFee());
                    dataList.add(data);
                }
                String fileName = String.format("transactions_%s.json",
                        LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")));
                String jsonArray = objectMapper.writeValueAsString(dataList);
                minIOService.archiveTransactionBatch(fileName, jsonArray);
                log.info("批量上传 {} 笔交易到 MinIO: {}", toUpload.size(), fileName);
            } catch (Exception e) {
                log.error("批量上传交易失败", e);
            }
        });
    }

    @PreDestroy
    public void flushAllBuffers() {
        flushBlockBuffer();
        flushTxBuffer();
    }

    @Override
    @CacheEvict(value = {"bitcoinStats", "blocksByTime", "transactionsByTime", "transactionDetail", "addressInfo"}, allEntries = true)
    public void clearAllCache() {
        log.info("清除所有比特币数据缓存");
    }
}