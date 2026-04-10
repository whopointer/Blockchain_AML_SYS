// EthereumDataServiceImpl.java
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
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.annotation.PreDestroy;
import java.io.BufferedWriter;
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

    private static final int MAX_QUERY_DAYS = 7;
    private static final int MAX_BLOCKS_PER_QUERY = 1000;
    private static final int MAX_TRANSACTIONS_PER_QUERY = 10000;
    private static final String CHAIN_ETH = "ETH";

    private static final String WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2";
    private static final String WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599";
    private static final String USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7";
    private static final String USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48";
    private static final String DAI  = "0x6b175474e89094c44da98b954eedeac495271d0f";
    private static final String ETH_NATIVE = "0x0000000000000000000000000000000000000000";

    @Autowired
    private io.minio.MinioClient minioClient;

    @Value("${minio.bucket-name}")
    private String minioBucketName;

    @Value("${minio.endpoint}")
    private String minioEndpoint;


    private final List<ChainBlock> blockBuffer = Collections.synchronizedList(new ArrayList<>());
    private final List<ChainTx> txBuffer = Collections.synchronizedList(new ArrayList<>());
    private static final int BLOCK_BATCH_SIZE = 100;   // 每100个区块合并上传
    private static final int TX_BATCH_SIZE = 5000;     // 每5000笔交易合并上传

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
                String jsonResponse = objectMapper.writeValueAsString(dataList);
                minIOService.archiveRawBigQueryResponse(queryType, jsonResponse);
            } catch (Exception e) {
                log.error("归档BigQuery结果失败: {}", queryType, e);
                uploadErrorToMinio("归档BigQuery结果失败", queryType, e);
            }
        });
    }

    private void archiveTransactionData(ChainTx tx) {
        if (!minioArchiveEnabled) return;
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
                uploadErrorToMinio("归档交易数据失败", tx.getTxHash(), e);
            }
        });
    }

    private void archiveBlockData(ChainBlock block) {
        if (!minioArchiveEnabled) return;
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
                uploadErrorToMinio("归档区块数据失败", "height=" + block.getHeight(), e);
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
                logEntry.put("service", "EthereumDataService");
                String logContent = objectMapper.writeValueAsString(logEntry);
                if (success) {
                    minIOService.archiveSyncLog("ETH", logContent);
                } else {
                    minIOService.archiveErrorLog(logContent);
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
    public ApiResponse<String> syncHistoricalData(LocalDate startDate, LocalDate endDate, Integer batchDays) {
        try {
            log.info("开始同步历史数据: {} 到 {}", startDate, endDate);
            archiveSyncLog("syncHistoricalData",
                    String.format("开始同步历史数据: %s 到 %s", startDate, endDate), true);

            if (startDate == null) startDate = LocalDate.of(2026, 1, 20);
            if (endDate == null) endDate = LocalDate.now().minusDays(1);
            if (batchDays == null || batchDays <= 0) batchDays = 1;
            if (batchDays > 30) batchDays = 30;

            int totalBlocks = 0;

            LocalDate currentDate = startDate;
            while (!currentDate.isAfter(endDate)) {
                LocalDate batchEndDate = currentDate.plusDays(batchDays - 1);
                if (batchEndDate.isAfter(endDate)) {
                    batchEndDate = endDate;
                }

                log.info("同步日期批次: {} 到 {}", currentDate, batchEndDate);

               int blocks = syncBlocksByDateRange(currentDate, batchEndDate);
               totalBlocks += blocks;

                for (LocalDate date = currentDate; !date.isAfter(batchEndDate); date = date.plusDays(1)) {
                    syncTwoHourlyTransactions(date);
                    for (int startHour = 0; startHour < 24; startHour += 2) {
                        syncTwoHourlyTokenTransfers(date, startHour, startHour + 2);
                    }
                }

                currentDate = batchEndDate.plusDays(1);
                Thread.sleep(1000);
            }

            archiveSyncLog("syncHistoricalData",
                    String.format("历史数据同步完成，共同步 %d 个区块", totalBlocks), true);
            return ApiResponse.success(String.format("历史数据同步完成，共同步 %d 个区块", totalBlocks), null);
        } catch (Exception e) {
            log.error("同步历史数据失败", e);
            uploadErrorToMinio("同步历史数据失败",
                    String.format("startDate=%s, endDate=%s, batchDays=%d", startDate, endDate, batchDays), e);
            archiveSyncLog("syncHistoricalData", "同步失败: " + e.getMessage(), false);
            return ApiResponse.error(500, "历史数据同步失败: " + e.getMessage());
        }
    }
    /**
     * 临时方法：仅将交易写入图数据库，不检查 MySQL，不写 MySQL，不归档 MinIO
     * 用于 Neo4j 清空后补全历史数据
     */
    private void syncTwoHourlyTransactionsOnlyGraph(LocalDate date) {
        for (int startHour = 20; startHour < 24; startHour += 2) {
            LocalDateTime start = date.atTime(startHour, 0, 0);
            LocalDateTime end = start.plusHours(2);
            String startTimestamp = start.format(BIGQUERY_TIMESTAMP_FORMAT);
            String endTimestamp = end.format(BIGQUERY_TIMESTAMP_FORMAT);
            String query = String.format(
                    "SELECT `hash`, `block_number`, `block_timestamp`, `from_address`, `to_address`, `value`, " +
                            "`gas_price`, `receipt_gas_used`, `gas`, `nonce`, `input`, `transaction_index`, " +
                            "`receipt_status` " +
                            "FROM `bigquery-public-data.crypto_ethereum.transactions` " +
                            "WHERE `block_timestamp` >= TIMESTAMP('%s') AND `block_timestamp` < TIMESTAMP('%s') " +
                            "ORDER BY `block_number`, `transaction_index`",
                    startTimestamp, endTimestamp
            );
            try {
                TableResult result = executeBigQuery(query);
                List<ChainTx> batch = new ArrayList<>(5000);
                int savedCount = 0;

                for (FieldValueList row : result.iterateAll()) {
                    ChainTx tx = mapToChainTxSingle(row);
                    batch.add(tx);
                    if (batch.size() >= 5000) {
                        // 直接调用图数据库批量保存，不经过 MySQL
                        graphService.saveTransactionsBatchToGraph(batch);
                        savedCount += batch.size();
                        batch.clear();
                    }
                }
                if (!batch.isEmpty()) {
                    graphService.saveTransactionsBatchToGraph(batch);
                    savedCount += batch.size();
                }
                log.info("日期 {} 时段 {}-{} 仅图数据库同步交易 {} 笔", date, startHour, startHour + 2, savedCount);
            } catch (Exception e) {
                log.error("仅图数据库同步交易失败: {} {}-{}", date, startHour, startHour + 2, e);
                // 不抛出异常，避免中断其他时段
            }
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
            if (minioArchiveEnabled) archiveBigQueryResult("blocks_query", result);

            List<ChainBlock> blocks = mapToChainBlocks(result);
            int savedCount = 0;
            for (ChainBlock block : blocks) {
                Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                if (!existingBlock.isPresent()) {
                    block.setChain(CHAIN_ETH);
                    chainBlockRepository.save(block);
                    addToBlockBuffer(block);  // 改为批量缓冲区
                    savedCount++;
                }
            }
            log.info("同步区块: {} 到 {}，新增 {} 个", startDate, endDate, savedCount);
            return savedCount;
        } catch (Exception e) {
            log.error("同步区块失败: {} - {}", startDate, endDate, e);
            uploadErrorToMinio("同步区块失败", String.format("startDate=%s, endDate=%s", startDate, endDate), e);
            return 0;
        }
    }

    private void syncTwoHourlyTransactions(LocalDate date) {
        for (int startHour = 0; startHour < 24; startHour += 2) {
            LocalDateTime start = date.atTime(startHour, 0, 0);
            LocalDateTime end = start.plusHours(2);
            String startTimestamp = start.format(BIGQUERY_TIMESTAMP_FORMAT);
            String endTimestamp = end.format(BIGQUERY_TIMESTAMP_FORMAT);
            String query = String.format(
                    "SELECT `hash`, `block_number`, `block_timestamp`, `from_address`, `to_address`, `value`, " +
                            "`gas_price`, `receipt_gas_used`, `gas`, `nonce`, `input`, `transaction_index`, " +
                            "`receipt_status` " +
                            "FROM `bigquery-public-data.crypto_ethereum.transactions` " +
                            "WHERE `block_timestamp` >= TIMESTAMP('%s') AND `block_timestamp` < TIMESTAMP('%s') " +
                            "ORDER BY `block_number`, `transaction_index`",
                    startTimestamp, endTimestamp
            );
            try {
                TableResult result = executeBigQuery(query);
                Set<String> existingTxHashes = chainTxRepository.findTxHashesByTimeRange(CHAIN_ETH, start, end);
                List<ChainTx> batch = new ArrayList<>(5000);
                int savedCount = 0;

                for (FieldValueList row : result.iterateAll()) {
                    ChainTx tx = mapToChainTxSingle(row);
                    if (!existingTxHashes.contains(tx.getTxHash())) {
                        batch.add(tx);
                        if (batch.size() >= 5000) {
                            saveTransactionsBatch(batch);
                            savedCount += batch.size();
                            batch.clear();
                        }
                    }
                }
                if (!batch.isEmpty()) {
                    saveTransactionsBatch(batch);
                    savedCount += batch.size();
                }
                log.info("日期 {} 时段 {}-{} 同步交易 {} 笔", date, startHour, startHour+2, savedCount);
            } catch (Exception e) {
                log.error("同步时段交易失败: {} {}-{}", date, startHour, startHour+2, e);
                uploadErrorToMinio("同步时段交易失败",
                        String.format("date=%s, startHour=%d", date, startHour), e);
            }
        }
    }

    private boolean syncTwoHourlyTokenTransfers(LocalDate date, int startHour, int endHour) {
        LocalDateTime start = date.atTime(startHour, 0, 0);
        LocalDateTime end = start.plusHours(2);
        String startTimestamp = start.format(BIGQUERY_TIMESTAMP_FORMAT);
        String endTimestamp = end.format(BIGQUERY_TIMESTAMP_FORMAT);

        String query = String.format(
                "SELECT `block_number`, `block_timestamp`, `transaction_hash`, `log_index`, `token_address`, " +
                        "`from_address`, `to_address`, `value` " +
                        "FROM `bigquery-public-data.crypto_ethereum.token_transfers` " +
                        "WHERE `block_timestamp` >= TIMESTAMP('%s') AND `block_timestamp` < TIMESTAMP('%s')",
                startTimestamp, endTimestamp
        );

        try {
            TableResult result = executeBigQuery(query);
            Set<String> existingKeys = chainTokenTransferRepository.findKeysByTimeRange(start, end);
            List<ChainTokenTransfer> batch = new ArrayList<>(5000);
            int savedCount = 0;

            for (FieldValueList row : result.iterateAll()) {
                ChainTokenTransfer tt = mapToChainTokenTransferSingle(row);
                String key = tt.getTransactionHash() + "#" + tt.getLogIndex();
                if (!existingKeys.contains(key)) {
                    batch.add(tt);
                    if (batch.size() >= 5000) {
                        chainTokenTransferRepository.saveAll(batch);
                        savedCount += batch.size();
                        batch.clear();
                    }
                }
            }
            if (!batch.isEmpty()) {
                chainTokenTransferRepository.saveAll(batch);
                savedCount += batch.size();
            }
            log.info("日期 {} 时段 {}-{} 同步代币转账 {} 条", date, startHour, endHour, savedCount);
            return true;
        } catch (Exception e) {
            log.error("同步时段代币转账失败: {} {}-{}", date, startHour, endHour, e);
            uploadErrorToMinio("同步时段代币转账失败",
                    String.format("date=%s, startHour=%d, endHour=%d", date, startHour, endHour), e);
            return false;
        }
    }

    private void saveTransactionsBatch(List<ChainTx> txs) {
        try {
            chainTxRepository.saveAll(txs);
            List<ChainTxInput> inputs = new ArrayList<>();
            List<ChainTxOutput> outputs = new ArrayList<>();
            for (ChainTx tx : txs) {
                ChainTxInput input = new ChainTxInput();
                input.setChain(CHAIN_ETH);
                input.setTransaction(tx);
                input.setInputIndex(0);
                input.setAddress(tx.getFromAddress());
                input.setValue(tx.getTotalInput() != null ? tx.getTotalInput() : BigDecimal.ZERO);
                inputs.add(input);

                if (tx.getToAddress() != null && !tx.getToAddress().isEmpty()) {
                    ChainTxOutput output = new ChainTxOutput();
                    output.setChain(CHAIN_ETH);
                    output.setTransaction(tx);
                    output.setOutputIndex(0);
                    output.setAddress(tx.getToAddress());
                    output.setValue(tx.getTotalOutput() != null ? tx.getTotalOutput() : BigDecimal.ZERO);
                    outputs.add(output);
                }
                addToTxBuffer(tx);  // 改为批量缓冲区
            }
            if (!inputs.isEmpty()) chainTxInputRepository.saveAll(inputs);
            if (!outputs.isEmpty()) chainTxOutputRepository.saveAll(outputs);

            if (graphService != null) {
                int graphBatchSize = 1000;
                for (int i = 0; i < txs.size(); i += graphBatchSize) {
                    List<ChainTx> subList = txs.subList(i, Math.min(i + graphBatchSize, txs.size()));
                    try {
                        graphService.saveTransactionsBatchToGraph(subList);
                    } catch (Exception e) {
                        log.error("图数据库批量保存失败，批次 {}-{}", i, i + subList.size(), e);
                        uploadErrorToMinio("图数据库批量保存失败",
                                String.format("batch start=%d, size=%d", i, subList.size()), e);
                    }
                }
            }
        } catch (Exception e) {
            log.error("批量保存交易失败", e);
            uploadErrorToMinio("批量保存交易失败", "batchSize=" + txs.size(), e);
        }
    }

    private ChainTx mapToChainTxSingle(FieldValueList row) {
        ChainTx tx = new ChainTx();
        tx.setChain(CHAIN_ETH);
        try {
            FieldValue valueField = row.get("value");
            BigInteger valueWei;
            if (!valueField.isNull()) {
                if (valueField.getAttribute() == FieldValue.Attribute.PRIMITIVE) {
                    try {
                        valueWei = BigInteger.valueOf(valueField.getLongValue());
                    } catch (Exception e) {
                        valueWei = new BigInteger(valueField.getStringValue());
                    }
                } else {
                    valueWei = new BigInteger(valueField.getStringValue());
                }
            } else {
                valueWei = BigInteger.ZERO;
            }
            tx.setValueWei(valueWei);
            BigDecimal valueEth = convertWeiToEth(new BigDecimal(valueWei));
            tx.setTotalOutput(valueEth);

            tx.setTxHash(row.get("hash").getStringValue());
            tx.setBlockHeight(row.get("block_number").getLongValue());

            String timestampStr = row.get("block_timestamp").getStringValue();
            tx.setBlockTime(parseBigQueryTimestamp(timestampStr));

            tx.setFromAddress(row.get("from_address").getStringValue());
            if (!row.get("to_address").isNull()) {
                tx.setToAddress(row.get("to_address").getStringValue());
            }

            if (!row.get("transaction_index").isNull()) {
                tx.setTxIndex((int) row.get("transaction_index").getLongValue());
            } else {
                tx.setTxIndex(0);
            }

            if (!row.get("receipt_status").isNull()) {
                long status = row.get("receipt_status").getLongValue();
                tx.setStatus(status == 1 ? "confirmed" : "failed");
            } else {
                tx.setStatus("confirmed");
            }

            if (!row.get("gas_price").isNull() && !row.get("receipt_gas_used").isNull()) {
                BigInteger gasPriceWei = new BigInteger(row.get("gas_price").getStringValue());
                BigInteger gasUsed = new BigInteger(row.get("receipt_gas_used").getStringValue());
                BigInteger feeWei = gasPriceWei.multiply(gasUsed);
                BigDecimal feeEth = convertWeiToEth(new BigDecimal(feeWei));
                tx.setFee(feeEth);
                tx.setTotalInput(valueEth.add(feeEth));
            }

            tx.setSizeBytes(0L);
            tx.setLocktime(0L);
        } catch (Exception e) {
            log.error("映射单条交易失败", e);
            uploadErrorToMinio("映射单条交易失败", "row=" + row.toString(), e);
        }
        return tx;
    }

    private ChainTokenTransfer mapToChainTokenTransferSingle(FieldValueList row) {
        ChainTokenTransfer tt = new ChainTokenTransfer();
        try {
            tt.setBlockNumber(row.get("block_number").getLongValue());
            tt.setBlockTimestamp(parseBigQueryTimestamp(row.get("block_timestamp").getStringValue()));
            tt.setTransactionHash(row.get("transaction_hash").getStringValue());
            tt.setLogIndex((int) row.get("log_index").getLongValue());
            tt.setTokenAddress(row.get("token_address").getStringValue());
            tt.setFromAddress(row.get("from_address").getStringValue());
            tt.setToAddress(row.get("to_address").getStringValue());

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
            tt.setValue(value.toString());
        } catch (Exception e) {
            log.error("映射单条代币转账失败", e);
            uploadErrorToMinio("映射单条代币转账失败", "row=" + row.toString(), e);
        }
        return tt;
    }

    // ============= 原有方法：测试图数据库连接 =============

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
            uploadErrorToMinio("测试图数据库连接失败", "", e);
            return ApiResponse.error(500, "测试图数据库连接失败: " + e.getMessage());
        }
    }

    private void saveTransactionInputOutput(ChainTx tx) {
        try {
            ChainTxInput input = new ChainTxInput();
            input.setChain(CHAIN_ETH);
            input.setTransaction(tx);
            input.setInputIndex(0);
            input.setAddress(tx.getFromAddress());
            input.setValue(tx.getTotalInput() != null ? tx.getTotalInput() : BigDecimal.ZERO);
            chainTxInputRepository.save(input);

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
            uploadErrorToMinio("保存交易输入输出失败", "txHash=" + tx.getTxHash(), e);
        }
    }

    // ============= 2. 获取最新数据 =============

    @Override
    @Transactional
    public ApiResponse<Map<String, Object>> getLatestData() {
        try {
            Map<String, Object> result = new HashMap<>();
            Long latestHeightInDB = getLatestBlockHeightFromDB();
            LocalDateTime latestTimeInDB = getLatestBlockTimeFromDB();
            result.put("db_latest_height", latestHeightInDB);
            result.put("db_latest_time", latestTimeInDB);

            String latestBlockQuery =
                    "SELECT MAX(`number`) as max_height, MAX(`timestamp`) as max_time " +
                            "FROM `bigquery-public-data.crypto_ethereum.blocks` " +
                            "WHERE DATE(`timestamp`) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)";
            TableResult bqResult = executeBigQuery(latestBlockQuery);
            if (minioArchiveEnabled) archiveBigQueryResult("latest_block_query", bqResult);

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

            Long latestHeightInDB = getLatestBlockHeightFromDB();
            if (latestHeightInDB == null) latestHeightInDB = 0L;

            log.info("开始同步最新数据，数据库最新高度: {}，同步 {} 个区块", latestHeightInDB, blocksToSync);
            archiveSyncLog("syncLatestData",
                    String.format("开始同步最新数据，数据库最新高度: %s，同步 %s 个区块", latestHeightInDB, blocksToSync), true);

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
            if (minioArchiveEnabled) archiveBigQueryResult("latest_blocks_query", blocksResult);

            List<ChainBlock> blocks = mapToChainBlocks(blocksResult);
            if (blocks.isEmpty()) {
                return ApiResponse.success("已是最新数据，无需同步", null);
            }

            int savedBlocks = 0;
            int savedTransactions = 0;
            for (ChainBlock block : blocks) {
                try {
                    Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                    if (!existingBlock.isPresent()) {
                        block.setChain(CHAIN_ETH);
                        chainBlockRepository.save(block);
                        addToBlockBuffer(block);  // 改为批量缓冲区
                        savedBlocks++;
                        savedTransactions += syncBlockTransactions(block.getHeight());
                    }
                } catch (Exception e) {
                    log.error("保存区块失败: height={}", block.getHeight(), e);
                    uploadErrorToMinio("保存区块失败", "height=" + block.getHeight(), e);
                }
            }

            archiveSyncLog("syncLatestData",
                    String.format("同步完成，新增 %d 个区块，%d 笔交易", savedBlocks, savedTransactions), true);
            return ApiResponse.success(String.format("同步完成，新增 %d 个区块，%d 笔交易",
                    savedBlocks, savedTransactions), null);
        } catch (Exception e) {
            log.error("同步最新数据失败", e);
            uploadErrorToMinio("同步最新数据失败", "blocksToSync=" + blocksToSync, e);
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
            if (minioArchiveEnabled) archiveBigQueryResult("block_transactions_query", result);

            List<ChainTx> transactions = mapToChainTxs(result);
            List<ChainTx> batchForGraph = new ArrayList<>();
            int savedCount = 0;
            for (ChainTx tx : transactions) {
                Optional<ChainTx> existingTx = chainTxRepository.findByChainAndTxHash(CHAIN_ETH, tx.getTxHash());
                if (!existingTx.isPresent()) {
                    tx.setChain(CHAIN_ETH);
                    ChainTx savedTx = chainTxRepository.save(tx);
                    savedCount++;
                    batchForGraph.add(savedTx);
                    addToTxBuffer(savedTx);  // 改为批量缓冲区
                    saveTransactionInputOutput(savedTx);
                }
            }
            if (!batchForGraph.isEmpty() && graphService != null) {
                try {
                    graphService.saveTransactionsToGraph(batchForGraph);
                } catch (Exception e) {
                    log.error("批量保存到图数据库失败", e);
                    uploadErrorToMinio("批量保存到图数据库失败", "blockHeight=" + blockHeight, e);
                }
            }
            log.debug("同步区块 {} 的 {} 笔交易", blockHeight, savedCount);
            return savedCount;
        } catch (Exception e) {
            log.error("同步区块交易失败: {}", blockHeight, e);
            uploadErrorToMinio("同步区块交易失败", "blockHeight=" + blockHeight, e);
            return 0;
        }
    }

    // ============= 3. 定时同步 =============

 //   @Scheduled(cron = "0 0 4 * * ?")
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
            if (minioArchiveEnabled) archiveBigQueryResult("scheduled_blocks_query", blocksResult);

            List<ChainBlock> blocks = mapToChainBlocks(blocksResult);
            if (!blocks.isEmpty()) {
                int savedBlocks = 0;
                for (ChainBlock block : blocks) {
                    Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                    if (!existingBlock.isPresent()) {
                        block.setChain(CHAIN_ETH);
                        chainBlockRepository.save(block);
                        addToBlockBuffer(block);  // 改为批量缓冲区
                        savedBlocks++;
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
            uploadErrorToMinio("定时同步失败", "", e);
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
            Page<ChainBlock> blocks = chainBlockRepository.findByChain(CHAIN_ETH, pageable);
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
            Page<ChainTx> transactions = chainTxRepository.findByChain(CHAIN_ETH, pageable);
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
            if (limit == null || limit <= 0) limit = 20;
            if (limit > 100) limit = 100;

            Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "blockHeight", "txIndex"));
            Page<ChainTx> transactions = chainTxRepository.findByFromAddressOrToAddress(CHAIN_ETH, address, pageable);
            return ApiResponse.success(transactions.getContent(), transactions.getTotalElements());
        } catch (Exception e) {
            log.error("查询地址交易失败: {}", address, e);
            uploadErrorToMinio("查询地址交易失败", "address=" + address, e);
            return ApiResponse.error(500, "查询地址交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressBalance(String address) {
        try {
            Map<String, Object> result = new HashMap<>();
            BigDecimal balance = calculateAddressBalance(address);
            Long txCount = chainTxRepository.countByFromAddressOrToAddress(CHAIN_ETH, address);
            result.put("address", address);
            result.put("balance", balance);
            result.put("transactionCount", txCount != null ? txCount : 0);
            result.put("chain", CHAIN_ETH);
            return ApiResponse.success(result, null);
        } catch (Exception e) {
            log.error("查询地址余额失败: {}", address, e);
            uploadErrorToMinio("查询地址余额失败", "address=" + address, e);
            return ApiResponse.error(500, "查询地址余额失败: " + e.getMessage());
        }
    }

    private BigDecimal calculateAddressBalance(String address) {
        try {
            BigDecimal received = chainTxRepository.sumTotalOutputByToAddress(CHAIN_ETH, address);
            if (received == null) received = BigDecimal.ZERO;
            BigDecimal sent = chainTxRepository.sumTotalInputByFromAddress(CHAIN_ETH, address);
            if (sent == null) sent = BigDecimal.ZERO;
            return received.subtract(sent);
        } catch (Exception e) {
            log.error("计算地址余额失败: {}", address, e);
            uploadErrorToMinio("计算地址余额失败", "address=" + address, e);
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
                Long blocksCount = chainBlockRepository.countByChainAndBlockTimeBetween(CHAIN_ETH, dayStart, dayEnd);
                Long transactionsCount = chainTxRepository.countByChainAndBlockTimeBetween(CHAIN_ETH, dayStart, dayEnd);
                dayStat.put("date", currentDate.toString());
                dayStat.put("blocks", blocksCount != null ? blocksCount : 0);
                dayStat.put("transactions", transactionsCount != null ? transactionsCount : 0);
                dailyStats.add(dayStat);
                currentDate = currentDate.plusDays(1);
            }
            return ApiResponse.success(dailyStats, (long) dailyStats.size());
        } catch (Exception e) {
            log.error("获取每日统计失败", e);
            uploadErrorToMinio("获取每日统计失败", "days=" + days, e);
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

            List<ChainBlock> blocks = getBlocksFromDatabase(startHeight, endHeight, limit);
            if (!blocks.isEmpty()) {
                return ApiResponse.success(blocks, (long) blocks.size());
            }

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
            if (minioArchiveEnabled) archiveBigQueryResult("get_blocks_query", result);

            List<ChainBlock> blocksFromBQ = mapToChainBlocks(result);
            for (ChainBlock block : blocksFromBQ) {
                Optional<ChainBlock> existingBlock = chainBlockRepository.findByChainAndHeight(CHAIN_ETH, block.getHeight());
                if (!existingBlock.isPresent()) {
                    block.setChain(CHAIN_ETH);
                    chainBlockRepository.save(block);
                    addToBlockBuffer(block);  // 改为批量缓冲区
                }
            }
            return ApiResponse.success(blocksFromBQ, (long) blocksFromBQ.size());
        } catch (Exception e) {
            log.error("获取区块数据失败", e);
            uploadErrorToMinio("获取区块数据失败", String.format("startHeight=%d, endHeight=%d, limit=%d", startHeight, endHeight, limit), e);
            return ApiResponse.error(500, "获取区块数据失败: " + e.getMessage());
        }
    }

    private List<ChainBlock> getBlocksFromDatabase(Long startHeight, Long endHeight, Integer limit) {
        try {
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
        } catch (Exception e) {
            log.error("从数据库获取区块失败", e);
            uploadErrorToMinio("从数据库获取区块失败", String.format("startHeight=%d, endHeight=%d, limit=%d", startHeight, endHeight, limit), e);
            return Collections.emptyList();
        }
    }

    @Override
    @Transactional
    public ApiResponse<List<ChainTx>> getTransactions(Long blockHeight, Integer limit, Integer offset) {
        try {
            if (limit == null || limit <= 0) limit = 100;
            if (limit > 1000) limit = 1000;
            if (offset == null || offset < 0) offset = 0;

            Pageable pageable = PageRequest.of(offset / limit, limit, Sort.by("txIndex").ascending());
            List<ChainTx> transactions = chainTxRepository.findByChainAndBlockHeight(CHAIN_ETH, blockHeight, pageable);
            if (!transactions.isEmpty()) {
                return ApiResponse.success(transactions, (long) transactions.size());
            }

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
            if (minioArchiveEnabled) archiveBigQueryResult("get_transactions_query", result);

            List<ChainTx> transactionsFromBQ = mapToChainTxs(result);
            for (ChainTx tx : transactionsFromBQ) {
                Optional<ChainTx> existingTx = chainTxRepository.findByChainAndTxHash(CHAIN_ETH, tx.getTxHash());
                if (!existingTx.isPresent()) {
                    tx.setChain(CHAIN_ETH);
                    ChainTx savedTx = chainTxRepository.save(tx);
                    addToTxBuffer(savedTx);  // 改为批量缓冲区
                    saveTransactionInputOutput(savedTx);
                }
            }
            return ApiResponse.success(transactionsFromBQ, (long) transactionsFromBQ.size());
        } catch (Exception e) {
            log.error("获取交易数据失败", e);
            uploadErrorToMinio("获取交易数据失败", "blockHeight=" + blockHeight + ", limit=" + limit + ", offset=" + offset, e);
            return ApiResponse.error(500, "获取交易数据失败: " + e.getMessage());
        }
    }

    // ============= 数据映射方法（批量） =============

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
                block.setBlockTime(parseBigQueryTimestamp(timestampStr));
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

    private List<ChainTx> mapToChainTxs(TableResult result) {
        List<ChainTx> transactions = new ArrayList<>();
        for (FieldValueList row : result.iterateAll()) {
            try {
                ChainTx tx = new ChainTx();
                tx.setChain(CHAIN_ETH);
                FieldValue valueField = row.get("value");
                BigInteger valueWei;
                if (!valueField.isNull()) {
                    if (valueField.getAttribute() == FieldValue.Attribute.PRIMITIVE) {
                        try {
                            valueWei = BigInteger.valueOf(valueField.getLongValue());
                        } catch (Exception e) {
                            valueWei = new BigInteger(valueField.getStringValue());
                        }
                    } else {
                        valueWei = new BigInteger(valueField.getStringValue());
                    }
                } else {
                    valueWei = BigInteger.ZERO;
                }
                tx.setValueWei(valueWei);
                BigDecimal valueEth = convertWeiToEth(new BigDecimal(valueWei));
                tx.setTotalOutput(valueEth);

                tx.setTxHash(row.get("hash").getStringValue());
                tx.setBlockHeight(row.get("block_number").getLongValue());

                String timestampStr = row.get("block_timestamp").getStringValue();
                tx.setBlockTime(parseBigQueryTimestamp(timestampStr));

                tx.setFromAddress(row.get("from_address").getStringValue());
                if (!row.get("to_address").isNull()) {
                    tx.setToAddress(row.get("to_address").getStringValue());
                }

                if (!row.get("transaction_index").isNull()) {
                    tx.setTxIndex((int) row.get("transaction_index").getLongValue());
                } else {
                    tx.setTxIndex(0);
                }

                if (!row.get("receipt_status").isNull()) {
                    long status = row.get("receipt_status").getLongValue();
                    tx.setStatus(status == 1 ? "confirmed" : "failed");
                } else {
                    tx.setStatus("confirmed");
                }

                if (!row.get("gas_price").isNull() && !row.get("receipt_gas_used").isNull()) {
                    BigInteger gasPriceWei = new BigInteger(row.get("gas_price").getStringValue());
                    BigInteger gasUsed = new BigInteger(row.get("receipt_gas_used").getStringValue());
                    BigInteger feeWei = gasPriceWei.multiply(gasUsed);
                    BigDecimal feeEth = convertWeiToEth(new BigDecimal(feeWei));
                    tx.setFee(feeEth);
                    tx.setTotalInput(valueEth.add(feeEth));
                }

                tx.setSizeBytes(0L);
                tx.setLocktime(0L);
                transactions.add(tx);
            } catch (Exception e) {
                log.error("映射交易数据失败", e);
                uploadErrorToMinio("映射交易数据失败", "", e);
            }
        }
        return transactions;
    }

    // ============= 异步地址探索方法 =============

    @Override
    @Async
    public CompletableFuture<ApiResponse<Map<String, Object>>> exploreAndExport(String taskId,
                                                                                List<String> sources,
                                                                                List<String> allowed,
                                                                                List<String> forbidden,
                                                                                LocalDateTime startTime,
                                                                                LocalDateTime endTime) {
        log.info("开始地址探索任务 [{}]：源地址={}, 允许列表={}, 禁止列表={}, 时间范围={} - {}",
                taskId, sources, allowed, forbidden, startTime, endTime);

        ExploreTaskStatus status = new ExploreTaskStatus();
        status.setTaskId(taskId);
        status.setStartTime(LocalDateTime.now());
        status.setStatus("RUNNING");
        taskStatusMap.put(taskId, status);

        try {
            Set<String> allTxHashes = exploreAddressNetwork(sources, allowed, forbidden, startTime, endTime, status);
            log.info("任务 [{}] 探索完成，共收集到 {} 笔交易哈希", taskId, allTxHashes.size());

            List<ChainTx> fullTxs = fetchTransactionsByHashes(allTxHashes, startTime, endTime);
            log.info("任务 [{}] 获取到 {} 笔完整原生交易", taskId, fullTxs.size());

            List<ChainTokenTransfer> tokenTransfers = fetchTokenTransfersByTxHashes(allTxHashes, startTime, endTime);
            log.info("任务 [{}] 获取到 {} 条代币转账事件", taskId, tokenTransfers.size());

            Long minBlock = null;
            Long maxBlock = null;
            for (ChainTx tx : fullTxs) {
                long h = tx.getBlockHeight();
                if (minBlock == null || h < minBlock) minBlock = h;
                if (maxBlock == null || h > maxBlock) maxBlock = h;
            }
            for (ChainTokenTransfer tt : tokenTransfers) {
                long h = tt.getBlockNumber();
                if (minBlock == null || h < minBlock) minBlock = h;
                if (maxBlock == null || h > maxBlock) maxBlock = h;
            }

            String startPrefix = (minBlock == null) ? "0" : String.valueOf(minBlock).substring(0, Math.min(3, String.valueOf(minBlock).length()));
            String endPrefix   = (maxBlock == null) ? "0" : String.valueOf(maxBlock).substring(0, Math.min(3, String.valueOf(maxBlock).length()));

            List<String> generatedFiles = generateTwoCsvFiles(fullTxs, tokenTransfers, minBlock, maxBlock);

            Map<String, Object> resultMap = new HashMap<>();
            resultMap.put("files", generatedFiles);
            resultMap.put("startBlock", startPrefix);
            resultMap.put("endBlock", endPrefix);

            status.setStatus("COMPLETED");
            status.setEndTime(LocalDateTime.now());
            status.setResult(generatedFiles);
            status.setMessage("成功生成 " + generatedFiles.size() + " 个CSV文件");

            return CompletableFuture.completedFuture(ApiResponse.success(resultMap, (long) generatedFiles.size()));
        } catch (Exception e) {
            log.error("任务 [{}] 执行失败", taskId, e);
            uploadErrorToMinio("探索任务失败", "taskId=" + taskId, e);
            status.setStatus("FAILED");
            status.setEndTime(LocalDateTime.now());
            status.setMessage("失败：" + e.getMessage());
            return CompletableFuture.completedFuture(ApiResponse.error(500, "探索任务失败: " + e.getMessage()));
        }
    }

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

            try {
                Set<String> neighbors = getNeighborsFromGraph(addr, startTime, endTime);
                Set<String> txHashes = getTransactionHashesFromGraph(addr, startTime, endTime);
                allTxHashes.addAll(txHashes);

                for (String neighbor : neighbors) {
                    if (explored.contains(neighbor) || toExplore.contains(neighbor)) continue;
                    boolean isForbidden = forbiddenSet.contains(neighbor);
                    boolean isAllowed = allowedSet.contains(neighbor);
                    if (!isForbidden || isAllowed) {
                        toExplore.add(neighbor);
                    }
                }
            } catch (Exception e) {
                log.error("探索地址网络失败: {}", addr, e);
                uploadErrorToMinio("探索地址网络失败", "address=" + addr, e);
            }

            if (explored.size() % 100 == 0) {
                log.info("探索进度：已探索 {} 个地址，累计收集 {} 个交易哈希", explored.size(), allTxHashes.size());
            }
        }
        log.info("地址探索完成：共探索 {} 个地址，收集 {} 个交易哈希", explored.size(), allTxHashes.size());
        return allTxHashes;
    }

    private Set<String> getNeighborsFromGraph(String address, LocalDateTime start, LocalDateTime end) {
        try {
            return graphService.getNeighborAddresses(address, start, end);
        } catch (Exception e) {
            log.warn("从Neo4j获取邻居地址失败，将回退到数据库: {}", e.getMessage());
            uploadErrorToMinio("从Neo4j获取邻居地址失败", "address=" + address, e);
            return Collections.emptySet();
        }
    }

    private Set<String> getTransactionHashesFromGraph(String address, LocalDateTime start, LocalDateTime end) {
        try {
            return graphService.getTransactionHashes(address, start, end);
        } catch (Exception e) {
            log.warn("从Neo4j获取交易哈希失败，将回退到数据库: {}", e.getMessage());
            uploadErrorToMinio("从Neo4j获取交易哈希失败", "address=" + address, e);
            return Collections.emptySet();
        }
    }

    private List<ChainTx> fetchTransactionsByAddressAndTime(String address, LocalDateTime start, LocalDateTime end) {
        try {
            Sort sort = Sort.by(Sort.Direction.ASC, "blockHeight", "txIndex");
            List<ChainTx> localTxs = chainTxRepository.findByAddressAndTimeRange(CHAIN_ETH, address, start, end, sort);
            log.info("从本地数据库查询到地址 {} 的交易数：{}", address, localTxs.size());
            return localTxs;
        } catch (Exception e) {
            log.error("从本地数据库查询交易失败", e);
            uploadErrorToMinio("从本地数据库查询交易失败", "address=" + address, e);
            return Collections.emptyList();
        }
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
            uploadErrorToMinio("从BigQuery拉取交易失败", "address=" + address, e);
            return Collections.emptyList();
        }
    }

    private List<ChainTx> fetchTransactionsByHashes(Set<String> txHashes, LocalDateTime start, LocalDateTime end) {
        if (txHashes.isEmpty()) return Collections.emptyList();
        try {
            List<ChainTx> localTxs = chainTxRepository.findByChainAndTxHashIn(CHAIN_ETH, new ArrayList<>(txHashes));
            return localTxs;
        } catch (Exception e) {
            log.error("根据哈希批量获取交易失败", e);
            uploadErrorToMinio("根据哈希批量获取交易失败", "txHashesCount=" + txHashes.size(), e);
            return Collections.emptyList();
        }
    }

    private List<ChainTokenTransfer> fetchTokenTransfersByTxHashes(Set<String> txHashes, LocalDateTime start, LocalDateTime end) {
        if (txHashes.isEmpty()) return Collections.emptyList();
        try {
            List<ChainTokenTransfer> localTTs = chainTokenTransferRepository.findByTransactionHashIn(new ArrayList<>(txHashes));
            log.info("从本地数据库获取到代币转账记录: {} 条", localTTs.size());
            return localTTs;
        } catch (Exception e) {
            log.error("批量获取代币转账失败", e);
            uploadErrorToMinio("批量获取代币转账失败", "txHashesCount=" + txHashes.size(), e);
            return Collections.emptyList();
        }
    }
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
                tt.setValue(value.toString());
                list.add(tt);
            } catch (Exception e) {
                log.error("映射代币转账失败", e);
                uploadErrorToMinio("映射代币转账失败", "", e);
            }
        }
        return list;
    }

    private void syncTokenTransfersByDate(LocalDate date) {
        String startTime = date.atStartOfDay().format(BIGQUERY_TIMESTAMP_FORMAT);
        String endTime = date.atTime(23, 59, 59).format(BIGQUERY_TIMESTAMP_FORMAT);
        String query = String.format(
                "SELECT block_number, block_timestamp, transaction_hash, log_index, token_address, " +
                        "from_address, to_address, value " +
                        "FROM `bigquery-public-data.crypto_ethereum.token_transfers` " +
                        "WHERE block_timestamp BETWEEN TIMESTAMP('%s') AND TIMESTAMP('%s')",
                startTime, endTime
        );

        try {
            TableResult result = executeBigQuery(query);
            List<ChainTokenTransfer> allTransfers = mapToChainTokenTransfers(result);
            if (allTransfers.isEmpty()) {
                log.info("日期 {} 无代币转账数据", date);
                return;
            }

            Set<String> existingKeys = getExistingTokenTransferKeys(allTransfers);
            List<ChainTokenTransfer> newTransfers = new ArrayList<>();
            for (ChainTokenTransfer tt : allTransfers) {
                String key = tt.getTransactionHash() + "#" + tt.getLogIndex();
                if (!existingKeys.contains(key)) {
                    newTransfers.add(tt);
                }
            }

            if (!newTransfers.isEmpty()) {
                chainTokenTransferRepository.saveAll(newTransfers);
                log.info("日期 {} 保存了 {} 条代币转账记录", date, newTransfers.size());
            } else {
                log.info("日期 {} 所有代币转账均已存在", date);
            }
        } catch (Exception e) {
            log.error("同步代币转账失败: {}", date, e);
            uploadErrorToMinio("同步代币转账失败", "date=" + date, e);
        }
    }

    private Set<String> getExistingTokenTransferKeys(List<ChainTokenTransfer> transfers) {
        if (transfers.isEmpty()) return Collections.emptySet();
        try {
            Set<String> txHashes = transfers.stream()
                    .map(ChainTokenTransfer::getTransactionHash)
                    .collect(Collectors.toSet());
            List<ChainTokenTransfer> existing = chainTokenTransferRepository.findByTransactionHashIn(new ArrayList<>(txHashes));
            Set<String> keys = new HashSet<>();
            for (ChainTokenTransfer tt : existing) {
                keys.add(tt.getTransactionHash() + "#" + tt.getLogIndex());
            }
            return keys;
        } catch (Exception e) {
            log.error("获取已存在代币转账keys失败", e);
            uploadErrorToMinio("获取已存在代币转账keys失败", "", e);
            return Collections.emptySet();
        }
    }

    // ============= 生成CSV并上传 =============

    private List<String> generateTwoCsvFiles(List<ChainTx> nativeTxs,
                                             List<ChainTokenTransfer> tokenTxs,
                                             Long minBlock,
                                             Long maxBlock) throws IOException {
        List<String> files = new ArrayList<>();

        String nativeFileName = String.format("native_%d_%d.csv", minBlock != null ? minBlock : 0, maxBlock != null ? maxBlock : 0);
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

        String tokenFileName = String.format("token_%d_%d.csv", minBlock != null ? minBlock : 0, maxBlock != null ? maxBlock : 0);
        Map<String, Integer> txIndexMap = new HashMap<>();
        for (ChainTx tx : nativeTxs) {
            txIndexMap.put(tx.getTxHash(), tx.getTxIndex());
        }

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

        for (String file : files) {
            try {
                minIOService.uploadFile(file);
            } catch (Exception e) {
                log.error("上传文件到MinIO失败: {}", file, e);
                uploadErrorToMinio("上传文件到MinIO失败", "file=" + file, e);
            }
        }
        for (String file : files) {
            new File(file).delete();
        }
        return files;
    }

    private String formatTimestamp(LocalDateTime time) {
        return time.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
    }

    // ============= 工具方法 =============

    private TableResult executeBigQuery(String query) throws InterruptedException {
        try {
            QueryJobConfiguration queryConfig = QueryJobConfiguration.newBuilder(query).build();
            return bigQuery.query(queryConfig);
        } catch (Exception e) {
            log.error("BigQuery 查询失败 - SQL: {}", query);
            uploadErrorToMinio("BigQuery查询失败", query, e);
            throw e;
        }
    }

    private Long getLatestBlockHeightFromDB() {
        try {
            return chainBlockRepository.findMaxHeight(CHAIN_ETH);
        } catch (Exception e) {
            log.error("获取最新区块高度失败", e);
            uploadErrorToMinio("获取最新区块高度失败", "", e);
            return null;
        }
    }

    private LocalDateTime getLatestBlockTimeFromDB() {
        try {
            return chainBlockRepository.findLatestBlockTime(CHAIN_ETH);
        } catch (Exception e) {
            log.error("获取最新区块时间失败", e);
            uploadErrorToMinio("获取最新区块时间失败", "", e);
            return null;
        }
    }

    private BigDecimal convertWeiToEth(BigDecimal wei) {
        return wei.divide(BigDecimal.valueOf(1_000_000_000_000_000_000L), 18, BigDecimal.ROUND_HALF_UP);
    }

    private LocalDateTime parseBigQueryTimestamp(String timestampStr) {
        try {
            if (timestampStr.matches("-?\\d+(\\.\\d+)?([Ee][+-]?\\d+)?")) {
                double timestampDouble = Double.parseDouble(timestampStr);
                long milliseconds = (long) (timestampDouble * 1000);
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(milliseconds), ZoneId.of("UTC"));
            } else {
                String cleaned = timestampStr.replace(" UTC", "").replace("+00:00", "").trim();
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
                        // continue
                    }
                }
                return LocalDateTime.parse(cleaned, DateTimeFormatter.ISO_DATE_TIME);
            }
        } catch (Exception e) {
            log.warn("解析时间戳失败: {}, 使用当前时间", timestampStr, e);
            uploadErrorToMinio("解析时间戳失败", "timestampStr=" + timestampStr, e);
            return LocalDateTime.now();
        }
    }

    // ============= 缓存清理 =============

    @Override
    @CacheEvict(value = {"blockchainStats", "blocksByTime", "transactionsByTime",
            "transactionDetail", "addressInfo"}, allEntries = true)
    public void clearAllCache() {
        log.info("清除所有缓存");
    }

    // ============= 接口方法实现 =============

    @Override
    public ApiResponse<Long> getBlockNumber() {
        try {
            Long latestHeight = getLatestBlockHeightFromDB();
            return ApiResponse.success(latestHeight, null);
        } catch (Exception e) {
            log.error("获取区块高度失败", e);
            uploadErrorToMinio("获取区块高度失败", "", e);
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
            uploadErrorToMinio("获取交易详情失败", "txHash=" + txHash, e);
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
            uploadErrorToMinio("按时间获取交易失败", "startTime=" + startTime + ", endTime=" + endTime, e);
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
            uploadErrorToMinio("按时间获取区块失败", "startTime=" + startTime + ", endTime=" + endTime, e);
            return ApiResponse.error(500, "按时间获取区块失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<String> exportCoreTokenPrices(Integer startBlockId, Integer endBlockId) {
        File tempFile = null;
        try {
            if (startBlockId == null || endBlockId == null) {
                return ApiResponse.error(400, "startBlockId and endBlockId are required");
            }
            if (startBlockId >= endBlockId) {
                return ApiResponse.error(400, "startBlockId must be less than endBlockId");
            }

            long startBlock = (long) startBlockId * 10000;
            long endBlock = (long) endBlockId * 10000;

            log.info("开始导出核心代币价格，block_id 范围：{} -> {}，对应区块号：[{}, {})",
                    startBlockId, endBlockId, startBlock, endBlock);

            String sql = buildPriceQuery(startBlock, endBlock);
            TableResult result = executeBigQuery(sql);
            if (minioArchiveEnabled) {
                archiveBigQueryResult("core_price_export", result);
            }

            Map<Long, BlockPriceData> priceDataMap = new LinkedHashMap<>();
            for (FieldValueList row : result.iterateAll()) {
                long blockId = row.get("block_id").getLongValue();
                String tokenAddr = row.get("target_token").getStringValue();
                String date = row.get("date_str").getStringValue();
                BigDecimal avgPrice = row.get("avg_price").getNumericValue();

                BlockPriceData data = priceDataMap.computeIfAbsent(blockId, k -> new BlockPriceData());
                data.date = date;
                if (tokenAddr.equalsIgnoreCase(WETH)) {
                    data.wethPrice = avgPrice;
                } else if (tokenAddr.equalsIgnoreCase(WBTC)) {
                    data.wbtcPrice = avgPrice;
                }
            }

            String fileName = String.format("core_prices_%d_%d.csv", startBlockId, endBlockId);
            tempFile = File.createTempFile("core_prices_", ".csv");
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile))) {
                writer.write("block_id,token_address,coin_id,date,price_usd\n");

                for (Map.Entry<Long, BlockPriceData> entry : priceDataMap.entrySet()) {
                    long blockId = entry.getKey();
                    BlockPriceData data = entry.getValue();
                    String date = data.date;

                    if (data.wethPrice != null) {
                        writer.write(String.format("%d,%s,%s,%s,%s\n",
                                blockId, WETH, WETH, date, data.wethPrice.toPlainString()));
                        writer.write(String.format("%d,%s,%s,%s,%s\n",
                                blockId, ETH_NATIVE, ETH_NATIVE, date, data.wethPrice.toPlainString()));
                    }
                    if (data.wbtcPrice != null) {
                        writer.write(String.format("%d,%s,%s,%s,%s\n",
                                blockId, WBTC, WBTC, date, data.wbtcPrice.toPlainString()));
                    }
                    for (String stableAddr : Arrays.asList(USDT, USDC, DAI)) {
                        writer.write(String.format("%d,%s,%s,%s,1.000000\n",
                                blockId, stableAddr, stableAddr, date));
                    }
                }
            }

            String objectName = "core_prices/" + fileName;
            try {
                minioClient.uploadObject(
                        io.minio.UploadObjectArgs.builder()
                                .bucket(minioBucketName)
                                .object(objectName)
                                .filename(tempFile.getAbsolutePath())
                                .build()
                );
                log.info("文件已上传至 MinIO: {}", objectName);
            } catch (Exception e) {
                log.error("上传文件至 MinIO 失败", e);
                uploadErrorToMinio("上传文件至 MinIO 失败", "fileName=" + fileName, e);
                throw new RuntimeException("上传文件失败: " + e.getMessage(), e);
            }

            String fileUrl = minioEndpoint + "/" + minioBucketName + "/" + objectName;
            log.info("导出完成，文件下载地址: {}", fileUrl);
            return ApiResponse.success(fileUrl, null);

        } catch (Exception e) {
            log.error("导出核心代币价格失败", e);
            uploadErrorToMinio("导出核心代币价格失败", "startBlockId=" + startBlockId + ", endBlockId=" + endBlockId, e);
            return ApiResponse.error(500, "导出失败: " + e.getMessage());
        } finally {
            if (tempFile != null && tempFile.exists()) {
                tempFile.delete();
            }
        }
    }

    private String buildPriceQuery(long startBlock, long endBlock) {
        return String.format(
                "WITH filtered AS (\n" +
                        "  SELECT transaction_hash, block_number, block_timestamp, LOWER(token_address) as addr, value \n" +
                        "  FROM `bigquery-public-data.crypto_ethereum.token_transfers` \n" +
                        "  WHERE block_number >= %d AND block_number < %d \n" +
                        "    AND LOWER(token_address) IN ('%s','%s','%s','%s','%s')\n" +
                        "),\n" +
                        "tx_summary AS (\n" +
                        "  SELECT \n" +
                        "    transaction_hash,\n" +
                        "    MAX(block_number) as block_number,\n" +
                        "    MAX(block_timestamp) as block_timestamp,\n" +
                        "    MAX(IF(addr IN ('%s','%s'), addr, NULL)) as target_token,\n" +
                        "    SUM(IF(addr IN ('%s','%s'), CAST(value AS NUMERIC), 0)) as target_raw_sum,\n" +
                        "    MAX(IF(addr IN ('%s','%s','%s'), addr, NULL)) as stable_token,\n" +
                        "    SUM(IF(addr IN ('%s','%s','%s'), CAST(value AS NUMERIC), 0)) as stable_raw_sum\n" +
                        "  FROM filtered\n" +
                        "  GROUP BY transaction_hash\n" +
                        "  HAVING target_token IS NOT NULL AND stable_token IS NOT NULL\n" +
                        "),\n" +
                        "transaction_prices AS (\n" +
                        "  SELECT \n" +
                        "    CAST(FLOOR(block_number / 10000) AS INT64) as block_id,\n" +
                        "    target_token,\n" +
                        "    DATE(block_timestamp) as dt,\n" +
                        "    SAFE_DIVIDE(\n" +
                        "      SAFE_DIVIDE(stable_raw_sum, POWER(10, IF(stable_token IN ('%s','%s'), 6, 18))),\n" +
                        "      SAFE_DIVIDE(target_raw_sum, POWER(10, IF(target_token = '%s', 8, 18)))\n" +
                        "    ) as p\n" +
                        "  FROM tx_summary\n" +
                        ")\n" +
                        "SELECT \n" +
                        "  block_id, \n" +
                        "  target_token, \n" +
                        "  FORMAT_DATE('%%Y-%%m-%%d', MIN(dt)) as date_str,\n" +
                        "  AVG(p) as avg_price\n" +
                        "FROM transaction_prices\n" +
                        "WHERE p IS NOT NULL AND (\n" +
                        "    (target_token = '%s' AND p BETWEEN 500 AND 10000) OR \n" +
                        "    (target_token = '%s' AND p BETWEEN 10000 AND 200000)\n" +
                        "  )\n" +
                        "GROUP BY block_id, target_token",
                startBlock, endBlock,
                WETH, WBTC, USDT, USDC, DAI,
                WETH, WBTC,
                WETH, WBTC,
                USDT, USDC, DAI,
                USDT, USDC, DAI,
                USDT, USDC, WBTC,
                WETH, WBTC
        );
    }

    private static class BlockPriceData {
        String date;
        BigDecimal wethPrice;
        BigDecimal wbtcPrice;
    }

    @Override
    public ApiResponse<String> exportBlocksToCsv(Long startHeight, Long endHeight) {
        return ApiResponse.success("从数据库导出功能待实现", null);
    }

    @Override
    public ApiResponse<String> exportTransactionsToCsv(LocalDateTime startTime, LocalDateTime endTime) {
        return ApiResponse.success("从数据库导出功能待实现", null);
    }

    // 批量缓冲区操作方法
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
                    data.put("from_address", tx.getFromAddress());
                    data.put("to_address", tx.getToAddress());
                    data.put("value", tx.getTotalOutput());
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
}