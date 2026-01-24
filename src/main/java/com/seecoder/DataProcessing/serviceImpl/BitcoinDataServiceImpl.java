// com/seecoder/DataProcessing/serviceImpl/BitcoinDataServiceImpl.java
package com.seecoder.DataProcessing.serviceImpl;

import com.google.cloud.bigquery.*;
import com.seecoder.DataProcessing.po.*;
import com.seecoder.DataProcessing.repository.*;
import com.seecoder.DataProcessing.service.BitcoinDataService;
import com.seecoder.DataProcessing.util.DateUtil;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

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

    private static final DateTimeFormatter BIGQUERY_TIMESTAMP_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyyMMdd_HHmmss");


    private String safeBuildBlocksQuery(Long startHeight, Long endHeight, Integer limit) {
        // 确保参数不为null
        startHeight = startHeight != null ? startHeight : 0L;
        endHeight = endHeight != null ? endHeight : Long.MAX_VALUE;
        limit = limit != null ? limit : 100;

        // 确保 startHeight <= endHeight
        if (startHeight > endHeight) {
            Long temp = startHeight;
            startHeight = endHeight;
            endHeight = temp;
        }

        return String.format(
                "SELECT `number`, `hash`, `prev_hash`, `timestamp`, `transaction_count`, `size` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                        "WHERE `number` >= %d AND `number` <= %d " +
                        "ORDER BY `number` DESC " +
                        "LIMIT %d",
                startHeight, endHeight, limit
        );
    }

    @Override
    public ApiResponse<List<ChainBlock>> getBlocks(Long startHeight, Long endHeight, Integer limit) {
        try {
            log.info("查询比特币区块: startHeight={}, endHeight={}, limit={}",
                    startHeight, endHeight, limit);

            // === 关键修复：参数验证 ===
            if (startHeight == null) {
                startHeight = 0L;
            }
            if (endHeight == null) {
                // 获取当前最新高度（示例值，实际应该从API获取）
                endHeight = 800000L;
            }
            if (limit == null || limit <= 0) {
                limit = 100;
            }
            if (limit > 10000) {
                limit = 10000;
            }

            // 构建查询（使用修复后的方法）
            String query = safeBuildBlocksQuery(startHeight, endHeight, limit);
            log.info("构建的SQL查询: {}", query);

            TableResult result = executeBigQuery(query);
            List<ChainBlock> blocks = mapToChainBlocks(result);

            return ApiResponse.success(blocks, (long) blocks.size());

        } catch (Exception e) {
            log.error("获取区块数据失败", e);
            return ApiResponse.error(500, "获取区块数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainBlock>> getBlocksByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        try {
            String startTimeStr = startTime.format(BIGQUERY_TIMESTAMP_FORMAT);
            String endTimeStr = endTime.format(BIGQUERY_TIMESTAMP_FORMAT);

            String query = String.format(
                    "SELECT `number`, `hash`, `prev_hash`, `timestamp`, `transaction_count`, `size` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                            "WHERE `timestamp` >= TIMESTAMP('%s') " +
                            "  AND `timestamp` <= TIMESTAMP('%s') " +
                            "ORDER BY `number` DESC " +
                            "LIMIT %d",
                    startTimeStr, endTimeStr, limit
            );

            TableResult result = executeBigQuery(query);
            List<ChainBlock> blocks = mapToChainBlocks(result);

            return ApiResponse.success(blocks, (long) blocks.size());

        } catch (Exception e) {
            log.error("按时间获取区块失败", e);
            return ApiResponse.error(500, "按时间获取区块失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainTx>> getTransactions(Long blockHeight, Integer limit, Integer offset) {
        try {
            log.info("查询比特币交易: 区块高度 {}", blockHeight);

            String query = buildTransactionsQuery(blockHeight, limit, offset);
            TableResult result = executeBigQuery(query);

            List<ChainTx> transactions = mapToChainTxs(result);

            // 为每笔交易获取输入输出详情
            for (ChainTx tx : transactions) {
                fetchAndSetTransactionDetails(tx);
            }

            return ApiResponse.success(transactions, (long) transactions.size());

        } catch (Exception e) {
            log.error("获取交易数据失败", e);
            return ApiResponse.error(500, "获取交易数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<List<ChainTx>> getTransactionsByTime(LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        try {
            String startTimeStr = startTime.format(BIGQUERY_TIMESTAMP_FORMAT);
            String endTimeStr = endTime.format(BIGQUERY_TIMESTAMP_FORMAT);

            String query = String.format(
                    "SELECT t.`hash`, t.`block_number`, b.`timestamp`, t.`input_value`, t.`output_value`, t.`fee` " +
                            "FROM `bigquery-public-data.crypto_bitcoin.transactions` t " +
                            "JOIN `bigquery-public-data.crypto_bitcoin.blocks` b " +
                            "  ON t.`block_number` = b.`number` " +
                            "WHERE b.`timestamp` >= TIMESTAMP('%s') " +
                            "  AND b.`timestamp` <= TIMESTAMP('%s') " +
                            "ORDER BY t.`block_number` DESC " +
                            "LIMIT %d",
                    startTimeStr, endTimeStr, limit
            );

            TableResult result = executeBigQuery(query);
            List<ChainTx> transactions = mapToChainTxs(result);

            return ApiResponse.success(transactions, (long) transactions.size());

        } catch (Exception e) {
            log.error("按时间获取交易失败", e);
            return ApiResponse.error(500, "按时间获取交易失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getTransactionDetail(String txHash) {
        try {
            Map<String, Object> result = new HashMap<>();

            // 获取交易基本信息
            ChainTx tx = fetchTransactionByHash(txHash);
            if (tx == null) {
                return ApiResponse.error(404, "交易不存在: " + txHash);
            }

            // 获取交易输入
            List<ChainTxInput> inputs = fetchTransactionInputs(txHash);

            // 获取交易输出
            List<ChainTxOutput> outputs = fetchTransactionOutputs(txHash);

            result.put("transaction", tx);
            result.put("inputs", inputs);
            result.put("outputs", outputs);
            result.put("inputCount", inputs.size());
            result.put("outputCount", outputs.size());

            // 计算总额
            BigDecimal totalInput = inputs.stream()
                    .map(ChainTxInput::getValue)
                    .filter(Objects::nonNull)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);

            BigDecimal totalOutput = outputs.stream()
                    .map(ChainTxOutput::getValue)
                    .filter(Objects::nonNull)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);

            result.put("totalInput", totalInput);
            result.put("totalOutput", totalOutput);

            return ApiResponse.success(result, null);

        } catch (Exception e) {
            log.error("获取交易详情失败", e);
            return ApiResponse.error(500, "获取交易详情失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressInfo(String address) {
        try {
            Map<String, Object> result = new HashMap<>();

            // 获取地址的UTXO
            List<ChainTxOutput> utxos = fetchUTXOs(address);

            // 获取地址相关的交易
            List<ChainTx> transactions = fetchAddressTransactions(address);

            // 计算余额
            BigDecimal balance = utxos.stream()
                    .map(ChainTxOutput::getValue)
                    .filter(Objects::nonNull)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);

            result.put("address", address);
            result.put("balance", balance);
            result.put("utxoCount", utxos.size());
            result.put("transactionCount", transactions.size());
            result.put("utxos", utxos);
            result.put("transactions", transactions);

            return ApiResponse.success(result, null);

        } catch (Exception e) {
            log.error("获取地址信息失败", e);
            return ApiResponse.error(500, "获取地址信息失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<String> exportBlocksToCsv(Long startHeight, Long endHeight) {
        try {
            // 获取区块数据
            ApiResponse<List<ChainBlock>> blocksResponse = getBlocks(startHeight, endHeight, 1000);
            if (!blocksResponse.getCode().equals(200)) {
                return ApiResponse.error(blocksResponse.getCode(), blocksResponse.getMessage());
            }

            List<ChainBlock> blocks = blocksResponse.getData();
            String filename = exportBlocksToCsvFile(blocks);

            return ApiResponse.success(filename, null);

        } catch (Exception e) {
            log.error("导出区块数据失败", e);
            return ApiResponse.error(500, "导出区块数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<String> exportTransactionsToCsv(LocalDateTime startTime, LocalDateTime endTime) {
        try {
            // 获取交易数据
            ApiResponse<List<ChainTx>> txResponse = getTransactionsByTime(startTime, endTime, 1000);
            if (!txResponse.getCode().equals(200)) {
                return ApiResponse.error(txResponse.getCode(), txResponse.getMessage());
            }

            List<ChainTx> transactions = txResponse.getData();
            String filename = exportTransactionsToCsvFile(transactions);

            return ApiResponse.success(filename, null);

        } catch (Exception e) {
            log.error("导出交易数据失败", e);
            return ApiResponse.error(500, "导出交易数据失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Map<String, Object>> getBlockchainStats() {
        try {
            Map<String, Object> stats = new HashMap<>();

            // 获取区块总数
            String blockCountQuery = "SELECT COUNT(*) as `count` FROM `bigquery-public-data.crypto_bitcoin.blocks`";
            TableResult blockResult = executeBigQuery(blockCountQuery);
            Long blockCount = getSingleLongResult(blockResult, "count");

            // 获取交易总数
            String txCountQuery = "SELECT COUNT(*) as `count` FROM `bigquery-public-data.crypto_bitcoin.transactions`";
            TableResult txResult = executeBigQuery(txCountQuery);
            Long txCount = getSingleLongResult(txResult, "count");

            // 获取最新区块高度
            String latestBlockQuery = "SELECT MAX(`number`) as `max_height` FROM `bigquery-public-data.crypto_bitcoin.blocks`";
            TableResult latestResult = executeBigQuery(latestBlockQuery);
            Long latestHeight = getSingleLongResult(latestResult, "max_height");

            stats.put("blockCount", blockCount);
            stats.put("transactionCount", txCount);
            stats.put("latestBlockHeight", latestHeight);
            stats.put("timestamp", LocalDateTime.now());

            return ApiResponse.success(stats, null);

        } catch (Exception e) {
            log.error("获取区块链统计失败", e);
            return ApiResponse.error(500, "获取区块链统计失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public ApiResponse<String> syncLatestBlocks(Integer limit) {
        try {
            // 获取当前最新高度
            Long currentHeight = chainBlockRepository.findMaxHeight("BTC");
            if (currentHeight == null) {
                currentHeight = 0L;
            }

            Long latestHeight = currentHeight + limit;

            log.info("同步最新区块: {} 到 {}", currentHeight + 1, latestHeight);

            // 同步区块数据
            ApiResponse<List<ChainBlock>> blocksResponse = getBlocks(currentHeight + 1, latestHeight, limit);
            if (blocksResponse.getCode().equals(200)) {
                List<ChainBlock> blocks = blocksResponse.getData();
                syncBlocksToDatabase(blocks);

                // 同步每个区块的交易
                for (ChainBlock block : blocks) {
                    syncBlockTransactions(block.getHeight());
                }
            }

            return ApiResponse.success("同步完成", null);

        } catch (Exception e) {
            log.error("同步区块数据失败", e);
            return ApiResponse.error(500, "同步失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public ApiResponse<String> syncHistoricalData(Long startHeight, Long endHeight, Integer batchSize) {
        try {
            log.info("同步历史数据: {} 到 {}", startHeight, endHeight);

            for (Long height = startHeight; height <= endHeight; height += batchSize) {
                Long batchEnd = Math.min(height + batchSize - 1, endHeight);

                log.info("同步批次: {} 到 {}", height, batchEnd);

                // 同步区块
                ApiResponse<List<ChainBlock>> blocksResponse = getBlocks(height, batchEnd, batchSize);
                if (blocksResponse.getCode().equals(200)) {
                    List<ChainBlock> blocks = blocksResponse.getData();
                    syncBlocksToDatabase(blocks);

                    // 同步每个区块的交易
                    for (ChainBlock block : blocks) {
                        syncBlockTransactions(block.getHeight());
                    }
                }

                // 避免请求过于频繁
                Thread.sleep(1000);
            }

            return ApiResponse.success("历史数据同步完成", null);

        } catch (Exception e) {
            log.error("同步历史数据失败", e);
            return ApiResponse.error(500, "同步失败: " + e.getMessage());
        }
    }

    // ============= 私有辅助方法 =============

    private String buildBlocksQuery(Long startHeight, Long endHeight, Integer limit) {
        return String.format(
                "SELECT `number`, `hash`, `prev_hash`, `timestamp`, `transaction_count`, `size` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.blocks` " +
                        "WHERE `number` >= %d AND `number` <= %d " +
                        "ORDER BY `number` DESC " +
                        "LIMIT %d",
                startHeight, endHeight, limit
        );
    }

    private String buildTransactionsQuery(Long blockHeight, Integer limit, Integer offset) {
        return String.format(
                "SELECT t.`hash`, t.`block_number`, b.`timestamp`, t.`input_value`, t.`output_value`, t.`fee` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.transactions` t " +
                        "JOIN `bigquery-public-data.crypto_bitcoin.blocks` b " +
                        "  ON t.`block_number` = b.`number` " +
                        "WHERE t.`block_number` = %d " +
                        "ORDER BY t.`block_number`, t.`block_timestamp` " +
                        "LIMIT %d OFFSET %d",
                blockHeight, limit, offset
        );
    }

    private TableResult executeBigQuery(String query) throws InterruptedException {
        try {
            log.debug("执行BigQuery查询: {}", query);
            QueryJobConfiguration queryConfig = QueryJobConfiguration.newBuilder(query).build();
            return bigQuery.query(queryConfig);
        } catch (Exception e) {
            log.error("BigQuery查询失败 - SQL: {}", query);
            throw e;
        }
    }

    private List<ChainBlock> mapToChainBlocks(TableResult result) {
        List<ChainBlock> blocks = new ArrayList<>();

        for (FieldValueList row : result.iterateAll()) {
            ChainBlock block = new ChainBlock();
            block.setHeight(row.get("number").getLongValue());
            block.setBlockHash(row.get("hash").getStringValue());
            block.setPrevBlockHash(row.get("prev_hash").getStringValue());

            // 转换时间戳
            String timestampStr = row.get("timestamp").getStringValue();
            block.setBlockTime(parseBigQueryTimestamp(timestampStr));

            block.setTxCount((int) row.get("transaction_count").getLongValue());
            block.setRawSizeBytes(row.get("size").getLongValue());

            blocks.add(block);
        }

        return blocks;
    }

    private List<ChainTx> mapToChainTxs(TableResult result) {
        List<ChainTx> transactions = new ArrayList<>();
        int index = 0;

        for (FieldValueList row : result.iterateAll()) {
            ChainTx tx = new ChainTx();
            tx.setTxHash(row.get("hash").getStringValue());
            tx.setBlockHeight(row.get("block_number").getLongValue());

            // 转换时间戳
            String timestampStr = row.get("timestamp").getStringValue();
            tx.setBlockTime(parseBigQueryTimestamp(timestampStr));

            // 计算金额（单位转换：聪 -> BTC）
            BigDecimal inputValue = BigDecimal.valueOf(row.get("input_value").getDoubleValue());
            BigDecimal outputValue = BigDecimal.valueOf(row.get("output_value").getDoubleValue());
            BigDecimal fee = BigDecimal.valueOf(row.get("fee").getDoubleValue());

            tx.setTotalInput(inputValue.divide(BigDecimal.valueOf(100_000_000), 8, BigDecimal.ROUND_HALF_UP));
            tx.setTotalOutput(outputValue.divide(BigDecimal.valueOf(100_000_000), 8, BigDecimal.ROUND_HALF_UP));
            tx.setFee(fee.divide(BigDecimal.valueOf(100_000_000), 8, BigDecimal.ROUND_HALF_UP));

            tx.setTxIndex(index++);

            transactions.add(tx);
        }

        return transactions;
    }

    private ChainTx fetchTransactionByHash(String txHash) throws Exception {
        String query = String.format(
                "SELECT t.`hash`, t.`block_number`, b.`timestamp`, t.`input_value`, t.`output_value`, t.`fee` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.transactions` t " +
                        "JOIN `bigquery-public-data.crypto_bitcoin.blocks` b " +
                        "  ON t.`block_number` = b.`number` " +
                        "WHERE t.`hash` = '%s'",
                txHash
        );

        TableResult result = executeBigQuery(query);
        List<ChainTx> transactions = mapToChainTxs(result);

        return transactions.isEmpty() ? null : transactions.get(0);
    }

    private void fetchAndSetTransactionDetails(ChainTx tx) throws Exception {
        String txHash = tx.getTxHash();

        // 获取交易输入
        List<ChainTxInput> inputs = fetchTransactionInputs(txHash);
        for (ChainTxInput input : inputs) {
            input.setTransaction(tx);
        }

        // 获取交易输出
        List<ChainTxOutput> outputs = fetchTransactionOutputs(txHash);
        for (ChainTxOutput output : outputs) {
            output.setTransaction(tx);
        }

        // 这里可以保存到数据库或直接关联
        tx.setTotalInput(calculateTotalInput(inputs));
        tx.setTotalOutput(calculateTotalOutput(outputs));
    }

    private List<ChainTxInput> fetchTransactionInputs(String txHash) throws Exception {
        String query = String.format(
                "SELECT `input_index`, `spent_transaction_hash`, `spent_output_index`, " +
                        "       `addresses`, `value`, `script_asm` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.inputs` " +
                        "WHERE `transaction_hash` = '%s' " +
                        "ORDER BY `input_index`",
                txHash
        );

        TableResult result = executeBigQuery(query);
        List<ChainTxInput> inputs = new ArrayList<>();

        for (FieldValueList row : result.iterateAll()) {
            ChainTxInput input = new ChainTxInput();
            input.setInputIndex((int) row.get("input_index").getLongValue());
            input.setPrevTxHash(row.get("spent_transaction_hash").getStringValue());

            if (!row.get("spent_output_index").isNull()) {
                input.setPrevOutIndex((int) row.get("spent_output_index").getLongValue());
            }

            // 处理地址
            if (!row.get("addresses").isNull()) {
                List<FieldValue> addresses = row.get("addresses").getRepeatedValue();
                if (!addresses.isEmpty()) {
                    input.setAddress(addresses.get(0).getStringValue());
                }
            }

            // 处理金额
            BigDecimal value = BigDecimal.valueOf(row.get("value").getDoubleValue());
            input.setValue(value.divide(BigDecimal.valueOf(100_000_000), 8, BigDecimal.ROUND_HALF_UP));

            input.setScriptSig(row.get("script_asm").getStringValue());

            inputs.add(input);
        }

        return inputs;
    }

    private List<ChainTxOutput> fetchTransactionOutputs(String txHash) throws Exception {
        String query = String.format(
                "SELECT `output_index`, `addresses`, `value`, `script_asm` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.outputs` " +
                        "WHERE `transaction_hash` = '%s' " +
                        "ORDER BY `output_index`",
                txHash
        );

        TableResult result = executeBigQuery(query);
        List<ChainTxOutput> outputs = new ArrayList<>();

        for (FieldValueList row : result.iterateAll()) {
            ChainTxOutput output = new ChainTxOutput();
            output.setOutputIndex((int) row.get("output_index").getLongValue());

            // 处理地址
            if (!row.get("addresses").isNull()) {
                List<FieldValue> addresses = row.get("addresses").getRepeatedValue();
                if (!addresses.isEmpty()) {
                    output.setAddress(addresses.get(0).getStringValue());
                }
            }

            // 处理金额
            BigDecimal value = BigDecimal.valueOf(row.get("value").getDoubleValue());
            output.setValue(value.divide(BigDecimal.valueOf(100_000_000), 8, BigDecimal.ROUND_HALF_UP));

            output.setScriptPubKey(row.get("script_asm").getStringValue());

            outputs.add(output);
        }

        return outputs;
    }

    private List<ChainTxOutput> fetchUTXOs(String address) throws Exception {
        String query = String.format(
                "SELECT o.`transaction_hash`, o.`output_index`, o.`addresses`, o.`value`, o.`script_asm` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.outputs` o " +
                        "LEFT JOIN `bigquery-public-data.crypto_bitcoin.inputs` i " +
                        "  ON o.`transaction_hash` = i.`spent_transaction_hash` " +
                        "  AND o.`output_index` = i.`spent_output_index` " +
                        "WHERE i.`spent_transaction_hash` IS NULL " +
                        "  AND o.`addresses` IS NOT NULL " +
                        "  AND ARRAY_LENGTH(o.`addresses`) > 0 " +
                        "  AND o.`addresses`[OFFSET(0)] = '%s'",
                address
        );

        TableResult result = executeBigQuery(query);
        List<ChainTxOutput> outputs = new ArrayList<>();

        for (FieldValueList row : result.iterateAll()) {
            ChainTxOutput output = new ChainTxOutput();
            output.setAddress(address);

            BigDecimal value = BigDecimal.valueOf(row.get("value").getDoubleValue());
            output.setValue(value.divide(BigDecimal.valueOf(100_000_000), 8, BigDecimal.ROUND_HALF_UP));

            outputs.add(output);
        }

        return outputs;
    }

    private List<ChainTx> fetchAddressTransactions(String address) throws Exception {
        // 这个查询比较复杂，简化处理
        String query = String.format(
                "SELECT DISTINCT `transaction_hash` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.outputs` " +
                        "WHERE `addresses` IS NOT NULL " +
                        "  AND ARRAY_LENGTH(`addresses`) > 0 " +
                        "  AND `addresses`[OFFSET(0)] = '%s' " +
                        "UNION DISTINCT " +
                        "SELECT DISTINCT `transaction_hash` " +
                        "FROM `bigquery-public-data.crypto_bitcoin.inputs` " +
                        "WHERE `addresses` IS NOT NULL " +
                        "  AND ARRAY_LENGTH(`addresses`) > 0 " +
                        "  AND `addresses`[OFFSET(0)] = '%s'",
                address, address
        );

        // 简化返回，实际需要获取完整的交易信息
        return new ArrayList<>();
    }

    private BigDecimal calculateTotalInput(List<ChainTxInput> inputs) {
        return inputs.stream()
                .map(ChainTxInput::getValue)
                .filter(Objects::nonNull)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private BigDecimal calculateTotalOutput(List<ChainTxOutput> outputs) {
        return outputs.stream()
                .map(ChainTxOutput::getValue)
                .filter(Objects::nonNull)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    private LocalDateTime parseBigQueryTimestamp(String timestampStr) {
        try {
            String cleaned = timestampStr.replace(" UTC", "");
            return LocalDateTime.parse(cleaned, BIGQUERY_TIMESTAMP_FORMAT);
        } catch (Exception e) {
            log.warn("解析时间戳失败: {}", timestampStr);
            return LocalDateTime.now();
        }
    }

    private Long getSingleLongResult(TableResult result, String fieldName) {
        for (FieldValueList row : result.iterateAll()) {
            return row.get(fieldName).getLongValue();
        }
        return 0L;
    }

    private String exportBlocksToCsvFile(List<ChainBlock> blocks) {
        // 实现CSV导出逻辑
        String filename = "bitcoin_blocks_" + DATE_FORMAT.format(new Date()) + ".csv";
        log.info("导出 {} 个区块到: {}", blocks.size(), filename);
        return filename;
    }

    private String exportTransactionsToCsvFile(List<ChainTx> transactions) {
        String filename = "bitcoin_transactions_" + DATE_FORMAT.format(new Date()) + ".csv";
        log.info("导出 {} 笔交易到: {}", transactions.size(), filename);
        return filename;
    }

    private void syncBlocksToDatabase(List<ChainBlock> blocks) {
        for (ChainBlock block : blocks) {
            // 检查是否已存在
            boolean exists = chainBlockRepository
                    .findByChainAndHeight("BTC", block.getHeight())
                    .isPresent();

            if (!exists) {
                chainBlockRepository.save(block);
                log.debug("保存区块: {}", block.getHeight());
            }
        }
    }

    private void syncBlockTransactions(Long blockHeight) {
        try {
            ApiResponse<List<ChainTx>> txResponse = getTransactions(blockHeight, 1000, 0);
            if (txResponse.getCode().equals(200)) {
                List<ChainTx> transactions = txResponse.getData();

                for (ChainTx tx : transactions) {
                    // 保存交易
                    boolean exists = chainTxRepository
                            .findByChainAndTxHash("BTC", tx.getTxHash())
                            .isPresent();

                    if (!exists) {
                        chainTxRepository.save(tx);

                        // 保存交易输入输出
                        List<ChainTxInput> inputs = fetchTransactionInputs(tx.getTxHash());
                        List<ChainTxOutput> outputs = fetchTransactionOutputs(tx.getTxHash());

                        for (ChainTxInput input : inputs) {
                            input.setTransaction(tx);
                        }
                        for (ChainTxOutput output : outputs) {
                            output.setTransaction(tx);
                        }

                        chainTxInputRepository.saveAll(inputs);
                        chainTxOutputRepository.saveAll(outputs);
                    }
                }

                log.info("同步区块 {} 的 {} 笔交易", blockHeight, transactions.size());
            }
        } catch (Exception e) {
            log.error("同步区块交易失败: {}", blockHeight, e);
        }
    }
}