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
    private static final int MAX_QUERY_BLOCKS = 1000;
    private static final int MAX_TRANSACTIONS_PER_QUERY = 10000;

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
                            rowData.put(fieldName, fieldValue.getValue());
                        }
                    }
                    dataList.add(rowData);
                }
                String json = objectMapper.writeValueAsString(dataList);
                minIOService.archiveRawBigQueryResponse(queryType, json);
            } catch (Exception e) {
                log.error("归档BigQuery结果失败: {}", queryType, e);
            }
        });
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
            }
        });
    }

    // ============= 1. 分块加载历史数据 =============

    @Override
    @Transactional
    public ApiResponse<String> syncHistoricalData(Long startHeight, Long endHeight, Integer batchSize) {
        try {
            log.info("开始同步历史数据: {} 到 {}", startHeight, endHeight);
            archiveSyncLog("syncHistoricalData",
                    String.format("开始同步历史数据: %d 到 %d", startHeight, endHeight), true);

            if (startHeight == null) startHeight = 0L;
            if (endHeight == null) endHeight = 800000L;
            if (batchSize == null || batchSize <= 0) batchSize = 100;
            if (batchSize > 1000) batchSize = 1000;

            int totalBlocks = 0;
            int totalTransactions = 0;

            for (Long from = startHeight; from <= endHeight; from += batchSize) {
                Long to = Math.min(from + batchSize - 1, endHeight);
                log.info("同步批次: {} 到 {}", from, to);

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

            archiveSyncLog("syncHistoricalData",
                    String.format("历史数据同步完成，共同步 %d 个区块，%d 笔交易", totalBlocks, totalTransactions), true);
            return ApiResponse.success(String.format("历史数据同步完成，共同步 %d 个区块，%d 笔交易", totalBlocks, totalTransactions), null);
        } catch (Exception e) {
            log.error("同步历史数据失败", e);
            archiveSyncLog("syncHistoricalData", "同步失败: " + e.getMessage(), false);
            return ApiResponse.error(500, "历史数据同步失败: " + e.getMessage());
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
                    archiveBlockData(block);
                    savedBlocks++;
                    savedTransactions += syncBlockTransactions(block.getHeight());
                }
            }

            archiveSyncLog("syncLatestData",
                    String.format("同步完成，新增 %d 个区块，%d 笔交易", savedBlocks, savedTransactions), true);
            return ApiResponse.success(String.format("同步完成，新增 %d 个区块，%d 笔交易", savedBlocks, savedTransactions), null);
        } catch (Exception e) {
            log.error("同步最新数据失败", e);
            archiveSyncLog("syncLatestData", "同步失败: " + e.getMessage(), false);
            return ApiResponse.error(500, "同步最新数据失败: " + e.getMessage());
        }
    }

    /**
     * 同步单个区块的所有交易，包括 inputs 和 outputs
     */
    private int syncBlockTransactions(Long blockHeight) {
        try {
            // 获取区块时间（用于分区过滤）
            Optional<ChainBlock> blockOpt = chainBlockRepository.findByChainAndHeight(CHAIN_BTC, blockHeight);
            if (!blockOpt.isPresent()) {
                log.warn("区块 {} 不在数据库中，跳过交易同步", blockHeight);
                return 0;
            }
            LocalDateTime blockTime = blockOpt.get().getBlockTime();
            LocalDate blockDate = blockTime.toLocalDate();
            String startDate = blockDate.toString();
            String endDate = blockDate.plusDays(1).toString();
            // 提取月份第一天，用于 block_timestamp_month 分区条件
            String yearMonthFirst = blockTime.format(DateTimeFormatter.ofPattern("yyyy-MM-01"));

            // 查询交易基本信息，包含 inputs 和 outputs 嵌套字段
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
                    archiveTransactionData(savedTx);

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

            // 保存到图数据库
            if (!batchForGraph.isEmpty() && graphService != null) {
                try {
                    graphService.saveBitcoinTransactionsToGraph(batchForGraph, inputsMap, outputsMap);
                } catch (Exception e) {
                    log.error("批量保存比特币交易到图数据库失败", e);
                }
            }

            log.debug("同步区块 {} 的 {} 笔交易", blockHeight, transactions.size());
            return transactions.size();
        } catch (Exception e) {
            log.error("同步区块交易失败: {}", blockHeight, e);
            return 0;
        }
    }

    /**
     * 解析 inputs 数组
     */
    private List<ChainTxInput> parseInputs(FieldValueList row, ChainTx tx) {
        List<ChainTxInput> inputs = new ArrayList<>();
        FieldValue inputsField = row.get("inputs");
        if (inputsField.isNull()) return inputs;

        for (FieldValue inputRecord : inputsField.getRepeatedValue()) {
            List<FieldValue> fields = inputRecord.getRecordValue();
            ChainTxInput input = new ChainTxInput();
            input.setChain(CHAIN_BTC);
            // 注意：此时 tx 尚未保存，ID 为空，但我们先设置关联，保存时再重新设置
            input.setTransaction(tx);

            // 根据 BigQuery schema 的字段顺序解析，以下索引基于实际查询结果调整
            // 一般顺序：index, spent_transaction_hash, spent_output_index, script_asm, script_hex, sequence, required_signatures, type, addresses, value
            // 索引从0开始
            if (fields.size() > 0 && !fields.get(0).isNull()) {
                input.setInputIndex((int) fields.get(0).getLongValue());
            }
            if (fields.size() > 1 && !fields.get(1).isNull()) {
                input.setPrevTxHash(fields.get(1).getStringValue());
            }
            if (fields.size() > 2 && !fields.get(2).isNull()) {
                input.setPrevOutIndex((int) fields.get(2).getLongValue());
            }
            // scriptSig：优先用 script_hex (索引4)，否则用 script_asm (索引3)
            if (fields.size() > 4 && !fields.get(4).isNull()) {
                input.setScriptSig(fields.get(4).getStringValue());
            } else if (fields.size() > 3 && !fields.get(3).isNull()) {
                input.setScriptSig(fields.get(3).getStringValue());
            }
            // addresses 数组 (索引8) 取第一个地址
            if (fields.size() > 8 && !fields.get(8).isNull()) {
                List<FieldValue> addresses = fields.get(8).getRepeatedValue();
                if (!addresses.isEmpty()) {
                    input.setAddress(addresses.get(0).getStringValue());
                }
            }
            // value (索引9) 单位是聪，转换为 BTC
            if (fields.size() > 9 && !fields.get(9).isNull()) {
                BigDecimal satoshi = BigDecimal.valueOf(fields.get(9).getDoubleValue());
                input.setValue(convertSatoshiToBtc(satoshi));
            }
            inputs.add(input);
        }
        return inputs;
    }

    /**
     * 解析 outputs 数组
     */
    private List<ChainTxOutput> parseOutputs(FieldValueList row, ChainTx tx) {
        List<ChainTxOutput> outputs = new ArrayList<>();
        FieldValue outputsField = row.get("outputs");
        if (outputsField.isNull()) return outputs;

        for (FieldValue outputRecord : outputsField.getRepeatedValue()) {
            List<FieldValue> fields = outputRecord.getRecordValue();
            ChainTxOutput output = new ChainTxOutput();
            output.setChain(CHAIN_BTC);
            output.setTransaction(tx);

            // 一般顺序：index, script_asm, script_hex, required_signatures, type, addresses, value
            if (fields.size() > 0 && !fields.get(0).isNull()) {
                output.setOutputIndex((int) fields.get(0).getLongValue());
            }
            // scriptPubKey：优先用 script_hex (索引2)，否则用 script_asm (索引1)
            if (fields.size() > 2 && !fields.get(2).isNull()) {
                output.setScriptPubKey(fields.get(2).getStringValue());
            } else if (fields.size() > 1 && !fields.get(1).isNull()) {
                output.setScriptPubKey(fields.get(1).getStringValue());
            }
            // addresses 数组 (索引5) 取第一个地址
            if (fields.size() > 5 && !fields.get(5).isNull()) {
                List<FieldValue> addresses = fields.get(5).getRepeatedValue();
                if (!addresses.isEmpty()) {
                    output.setAddress(addresses.get(0).getStringValue());
                }
            }
            // value (索引6) 单位是聪
            if (fields.size() > 6 && !fields.get(6).isNull()) {
                BigDecimal satoshi = BigDecimal.valueOf(fields.get(6).getDoubleValue());
                output.setValue(convertSatoshiToBtc(satoshi));
            }
            outputs.add(output);
        }
        return outputs;
    }

    /**
     * 将 BigQuery 行映射为 ChainTx 基本对象（不含 inputs/outputs）
     */
    private ChainTx mapToChainTx(FieldValueList row) {
        ChainTx tx = new ChainTx();
        tx.setTxHash(row.get("hash").getStringValue());
        tx.setBlockHeight(row.get("block_number").getLongValue());
        String ts = row.get("block_timestamp").getStringValue();
        tx.setBlockTime(parseBigQueryTimestamp(ts));

        // 安全处理可能为 null 的数值字段
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
        return tx;
    }

    // ============= 3. 定时同步 =============

    @Scheduled(fixedDelay = 300000)
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
                        archiveBlockData(block);
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
            return ApiResponse.error(500, "查询地址余额失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressInfo(String address) {
        try {
            Map<String, Object> info = new HashMap<>();
            info.put("address", address);
            info.put("balance", getAddressBalance(address).getData());
            // 最近交易
            Pageable pageable = PageRequest.of(0, 10, Sort.by(Sort.Direction.DESC, "blockTime"));
            Page<ChainTx> recentTxs = chainTxRepository.findByFromAddressOrToAddress(CHAIN_BTC, address, pageable);
            info.put("recentTransactions", recentTxs.getContent());
            info.put("transactionCount", recentTxs.getTotalElements());
            return ApiResponse.success(info, null);
        } catch (Exception e) {
            log.error("查询地址信息失败", e);
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
            if (limit > MAX_QUERY_BLOCKS) limit = MAX_QUERY_BLOCKS;

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
                    archiveBlockData(block);
                }
            }
            return ApiResponse.success(blocks, (long) blocks.size());
        } catch (Exception e) {
            log.error("获取区块数据失败", e);
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

            // 从数据库获取区块时间（用于分区过滤）
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

                // 保存到数据库
                if (!chainTxRepository.findByChainAndTxHash(CHAIN_BTC, tx.getTxHash()).isPresent()) {
                    tx.setChain(CHAIN_BTC);
                    ChainTx savedTx = chainTxRepository.save(tx);
                    archiveTransactionData(savedTx);

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
                // blocks 表中无前一区块哈希字段
                String ts = row.get("timestamp").getStringValue();
                block.setBlockTime(parseBigQueryTimestamp(ts));
                block.setTxCount((int) row.get("transaction_count").getLongValue());
                block.setRawSizeBytes(row.get("size").getLongValue());
                blocks.add(block);
            } catch (Exception e) {
                log.error("映射区块数据失败", e);
            }
        }
        return blocks;
    }

    private BigDecimal convertSatoshiToBtc(BigDecimal satoshi) {
        return satoshi.divide(BigDecimal.valueOf(100_000_000), 8, RoundingMode.HALF_UP);
    }

    private TableResult executeBigQuery(String query) throws InterruptedException {
        try {
            log.debug("执行BigQuery查询: {}", query);
            QueryJobConfiguration config = QueryJobConfiguration.newBuilder(query).build();
            return bigQuery.query(config);
        } catch (Exception e) {
            log.error("BigQuery查询失败 - SQL: {}", query);
            throw e;
        }
    }

    /**
     * 增强的时间戳解析，支持：
     * - 科学计数法（如 1.619004665E9）
     * - 带毫秒的日期时间（如 2026-02-26 20:11:23.032）
     * - 不带毫秒的日期时间（如 2026-02-26 20:11:23）
     * - ISO 格式（如 2026-02-26T20:11:23）
     */
    private LocalDateTime parseBigQueryTimestamp(String timestampStr) {
        try {
            // 去除前后空格和 "UTC" 后缀
            String cleaned = timestampStr.trim().replace(" UTC", "");

            // 匹配数字（整数、小数、科学计数法）
            if (cleaned.matches("-?\\d+(\\.\\d+)?([Ee][+-]?\\d+)?")) {
                double seconds = Double.parseDouble(cleaned);
                long millis = (long) (seconds * 1000);
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(millis), ZoneId.of("UTC"));
            } else {
                // 尝试带毫秒的格式：yyyy-MM-dd HH:mm:ss.SSS
                try {
                    DateTimeFormatter formatterWithMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
                    return LocalDateTime.parse(cleaned, formatterWithMillis);
                } catch (Exception e1) {
                    // 尝试不带毫秒的格式：yyyy-MM-dd HH:mm:ss
                    try {
                        DateTimeFormatter formatterWithoutMillis = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
                        return LocalDateTime.parse(cleaned, formatterWithoutMillis);
                    } catch (Exception e2) {
                        // 尝试 ISO 格式：yyyy-MM-dd'T'HH:mm:ss
                        DateTimeFormatter isoFormatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME;
                        return LocalDateTime.parse(cleaned, isoFormatter);
                    }
                }
            }
        } catch (Exception e) {
            log.warn("解析时间戳失败: {}, 使用当前时间", timestampStr, e);
            return LocalDateTime.now();
        }
    }

    @Override
    @CacheEvict(value = {"bitcoinStats", "blocksByTime", "transactionsByTime", "transactionDetail", "addressInfo"}, allEntries = true)
    public void clearAllCache() {
        log.info("清除所有比特币数据缓存");
    }
}