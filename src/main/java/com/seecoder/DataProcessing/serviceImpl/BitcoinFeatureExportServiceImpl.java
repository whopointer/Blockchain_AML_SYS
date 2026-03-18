package com.seecoder.DataProcessing.serviceImpl;

import com.opencsv.CSVWriter;
import com.seecoder.DataProcessing.po.*;
import com.seecoder.DataProcessing.repository.*;
import com.seecoder.DataProcessing.service.FeatureExportService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphAddressService;
import com.seecoder.DataProcessing.util.FeatureUtils;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

@Slf4j
@Service("bitcoinFeatureExportService")
public class BitcoinFeatureExportServiceImpl implements FeatureExportService {

    private static final String CHAIN_BTC = "BTC";
    private static final int PAGE_SIZE = 500;

    @Autowired
    private ChainTxRepository chainTxRepository;
    @Autowired
    private ChainTxInputRepository chainTxInputRepository;
    @Autowired
    private ChainTxOutputRepository chainTxOutputRepository;
    @Autowired(required = false)
    private GraphAddressService graphAddressService; // 可能为 null

    @Override
    @Transactional(readOnly = true)
    public ApiResponse<String> exportFeatures(String chain,
                                              Long startHeight, Long endHeight,
                                              LocalDateTime startTime, LocalDateTime endTime,
                                              String filePath) {
        if (!CHAIN_BTC.equals(chain)) {
            return ApiResponse.error(400, "不支持的链: " + chain);
        }

        // 参数处理
        if (startHeight == null) startHeight = 0L;
        if (endHeight == null) endHeight = Long.MAX_VALUE;

        // 生成默认文件路径
        if (filePath == null || filePath.trim().isEmpty()) {
            String baseDir = System.getProperty("user.dir") + File.separator + "src" + File.separator + "data" + File.separator;
            File dir = new File(baseDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            filePath = baseDir + chain + "_" + startHeight + "_" + endHeight + "_" + timestamp + ".csv";
        }

        // 确保目标文件的父目录存在
        File targetFile = new File(filePath);
        File parentDir = targetFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        // ========== 修改点：预计算全局地址复用次数 ==========
        Map<String, Integer> globalAddressCount = preComputeAddressCount();
        // =================================================

        long totalTxs = 0;
        int page = 0;

        try (CSVWriter writer = new CSVWriter(new FileWriter(targetFile))) {
            writer.writeNext(TransactionFeatures.getCsvHeader());

            while (true) {
                Pageable pageable = PageRequest.of(page, PAGE_SIZE,
                        Sort.by(Sort.Direction.ASC, "blockHeight", "txIndex"));
                Page<ChainTx> txPage = chainTxRepository.findByChainAndBlockHeightBetween(
                        CHAIN_BTC, startHeight, endHeight, pageable);
                if (txPage.isEmpty()) break;

                List<ChainTx> txs = txPage.getContent();
                log.info("处理第 {} 批，共 {} 笔交易", page + 1, txs.size());

                // 预加载inputs/outputs
                List<Long> txIds = txs.stream().map(ChainTx::getId).collect(Collectors.toList());
                List<ChainTxInput> allInputs = chainTxInputRepository.findByTransactionIdIn(txIds);
                List<ChainTxOutput> allOutputs = chainTxOutputRepository.findByTransactionIdIn(txIds);

                Map<Long, List<ChainTxInput>> inputsMap = allInputs.stream()
                        .collect(Collectors.groupingBy(in -> in.getTransaction().getId()));
                Map<Long, List<ChainTxOutput>> outputsMap = allOutputs.stream()
                        .collect(Collectors.groupingBy(out -> out.getTransaction().getId()));

                // 收集上游交易哈希
                Set<String> prevTxHashes = allInputs.stream()
                        .map(ChainTxInput::getPrevTxHash)
                        .filter(Objects::nonNull)
                        .collect(Collectors.toSet());
                Map<String, ChainTx> prevTxMap = loadPrevTxs(prevTxHashes);

                // 预加载上游交易的输入输出数量
                Map<String, Integer> prevTxInputCount = loadInputCounts(prevTxMap.keySet());
                Map<String, Integer> prevTxOutputCount = loadOutputCounts(prevTxMap.keySet());

                // 最大跳数限制，可根据需要调整
                int maxHopsForCycle = 5;

                for (ChainTx tx : txs) {
                    TransactionFeatures features = new TransactionFeatures();
                    List<ChainTxInput> inputs = inputsMap.getOrDefault(tx.getId(), Collections.emptyList());
                    List<ChainTxOutput> outputs = outputsMap.getOrDefault(tx.getId(), Collections.emptyList());

                    // 基础特征
                    features.setNumInputs(inputs.size());
                    features.setNumOutputs(outputs.size());
                    features.setTxFee(FeatureUtils.logTransform(tx.getFeeAsDouble()));
                    features.setTxSizeBytes(tx.getSizeBytes() != null ? tx.getSizeBytes() : 0L);
                    features.setFeeRate(calculateFeeRate(tx, features.getTxSizeBytes()));
                    features.setTotalInputAmt(FeatureUtils.logTransform(tx.getTotalInputAsDouble()));
                    features.setTotalOutputAmt(FeatureUtils.logTransform(tx.getTotalOutputAsDouble()));
                    features.setOutputStdDev(FeatureUtils.outputStdDev(outputs));
                    features.setLocktime(tx.getLocktime() != null ? tx.getLocktime() : 0L);
                    features.setIsCoinbase(isCoinbase(inputs) ? 1 : 0);

                    // 邻居特征
                    if (!inputs.isEmpty()) {
                        List<ChainTx> prevTxs = inputs.stream()
                                .map(in -> prevTxMap.get(in.getPrevTxHash()))
                                .filter(Objects::nonNull)
                                .distinct()
                                .collect(Collectors.toList());

                        if (!prevTxs.isEmpty()) {
                            features.setNeighborInAmtMean(FeatureUtils.logTransform(
                                    FeatureUtils.mean(prevTxs, ChainTx::getTotalInputAsDouble)));
                            features.setNeighborInAmtStd(FeatureUtils.logTransform(
                                    FeatureUtils.stdDev(prevTxs, ChainTx::getTotalInputAsDouble)));
                            features.setNeighborInFeeMean(FeatureUtils.logTransform(
                                    FeatureUtils.mean(prevTxs, ChainTx::getFeeAsDouble)));

                            double avgInputCount = prevTxs.stream()
                                    .mapToInt(p -> prevTxInputCount.getOrDefault(p.getTxHash(), 0))
                                    .average().orElse(0.0);
                            double avgOutputCount = prevTxs.stream()
                                    .mapToInt(p -> prevTxOutputCount.getOrDefault(p.getTxHash(), 0))
                                    .average().orElse(0.0);
                            features.setNeighborInDegreeMean(avgInputCount);
                            features.setNeighborOutDegreeMean(avgOutputCount);

                            List<LocalDateTime> prevTimes = prevTxs.stream()
                                    .map(ChainTx::getBlockTime)
                                    .collect(Collectors.toList());
                            features.setNeighborTimeSpan(FeatureUtils.meanTimeSpan(tx.getBlockTime(), prevTimes));
                        } else {
                            features.setNeighborInAmtMean(0.0);
                            features.setNeighborInAmtStd(0.0);
                            features.setNeighborInFeeMean(0.0);
                            features.setNeighborInDegreeMean(0.0);
                            features.setNeighborOutDegreeMean(0.0);
                            features.setNeighborTimeSpan(0.0);
                        }
                    } else {
                        features.setNeighborInAmtMean(0.0);
                        features.setNeighborInAmtStd(0.0);
                        features.setNeighborInFeeMean(0.0);
                        features.setNeighborInDegreeMean(0.0);
                        features.setNeighborOutDegreeMean(0.0);
                        features.setNeighborTimeSpan(0.0);
                    }

                    // 币天销毁（小数天数）
                    features.setCoindaysDestroyed(calculateCDD(tx, inputs, prevTxMap));

                    // round_value_ratio
                    features.setRoundValueRatio(FeatureUtils.roundValueRatio(outputs));

                    // ========== 修改点：使用全局地址计数 ==========
                    Set<String> addressesInTx = new HashSet<>();
                    inputs.forEach(in -> { if (in.getAddress() != null) addressesInTx.add(in.getAddress()); });
                    outputs.forEach(out -> { if (out.getAddress() != null) addressesInTx.add(out.getAddress()); });
                    int maxReuse = addressesInTx.stream()
                            .mapToInt(addr -> globalAddressCount.getOrDefault(addr, 0))
                            .max().orElse(0);
                    features.setAddressReuseCount(maxReuse);
                    // ============================================

                    // 循环标志
                    int loopFlag = 0;
                    if (graphAddressService != null) {
                        try {
                            List<String> inputAddrs = inputs.stream()
                                    .map(ChainTxInput::getAddress)
                                    .filter(Objects::nonNull)
                                    .distinct()
                                    .collect(Collectors.toList());
                            List<String> outputAddrs = outputs.stream()
                                    .map(ChainTxOutput::getAddress)
                                    .filter(Objects::nonNull)
                                    .distinct()
                                    .collect(Collectors.toList());
                            if (!inputAddrs.isEmpty() && !outputAddrs.isEmpty()) {
                                boolean hasCycle = graphAddressService.existsCycle(inputAddrs, outputAddrs, maxHopsForCycle);
                                loopFlag = hasCycle ? 1 : 0;
                            }
                        } catch (Exception e) {
                            log.warn("检测循环异常，忽略该特征: txHash={}", tx.getTxHash(), e);
                        }
                    }
                    features.setLoopFlag(loopFlag);

                    // 扇入扇出比
                    features.setFanInRatio((double) inputs.size() / (outputs.size() + 1));

                    writer.writeNext(features.toCsvRow());
                    totalTxs++;
                }
                page++;
            }
            log.info("比特币特征导出完成，共 {} 笔交易", totalTxs);
            String resultMsg = "导出成功，共 " + totalTxs + " 笔交易，文件保存至: " + targetFile.getAbsolutePath();
            return ApiResponse.success(resultMsg, totalTxs);
        } catch (IOException e) {
            log.error("导出CSV失败", e);
            return ApiResponse.error(500, "导出失败: " + e.getMessage());
        }
    }

    // ========== 新增：全局地址计数预计算 ==========
    private Map<String, Integer> preComputeAddressCount() {
        Map<String, Integer> countMap = new HashMap<>();
        // 统计输入地址
        List<Object[]> inputCounts = chainTxInputRepository.countGroupByAddress(CHAIN_BTC);
        for (Object[] row : inputCounts) {
            String addr = (String) row[0];
            Number cnt = (Number) row[1];
            countMap.put(addr, cnt.intValue());
        }
        // 统计输出地址
        List<Object[]> outputCounts = chainTxOutputRepository.countGroupByAddress(CHAIN_BTC);
        for (Object[] row : outputCounts) {
            String addr = (String) row[0];
            Number cnt = (Number) row[1];
            countMap.merge(addr, cnt.intValue(), Integer::sum);
        }
        return countMap;
    }
    // =============================================

    private Double calculateFeeRate(ChainTx tx, Long sizeBytes) {
        if (sizeBytes == null || sizeBytes == 0) return 0.0;
        return tx.getFeeAsDouble() / sizeBytes;
    }

    private boolean isCoinbase(List<ChainTxInput> inputs) {
        return inputs.size() == 1 && inputs.get(0).getPrevTxHash() == null;
    }

    private Map<String, ChainTx> loadPrevTxs(Set<String> prevTxHashes) {
        if (prevTxHashes.isEmpty()) return Collections.emptyMap();
        List<ChainTx> txs = chainTxRepository.findByTxHashIn(new ArrayList<>(prevTxHashes));
        return txs.stream().collect(Collectors.toMap(ChainTx::getTxHash, Function.identity()));
    }

    private Map<String, Integer> loadInputCounts(Set<String> txHashes) {
        if (txHashes.isEmpty()) return Collections.emptyMap();
        List<Object[]> results = chainTxInputRepository.countInputsByTxHashIn(new ArrayList<>(txHashes));
        return results.stream().collect(Collectors.toMap(
                arr -> (String) arr[0],
                arr -> ((Number) arr[1]).intValue()
        ));
    }

    private Map<String, Integer> loadOutputCounts(Set<String> txHashes) {
        if (txHashes.isEmpty()) return Collections.emptyMap();
        List<Object[]> results = chainTxOutputRepository.countOutputsByTxHashIn(new ArrayList<>(txHashes));
        return results.stream().collect(Collectors.toMap(
                arr -> (String) arr[0],
                arr -> ((Number) arr[1]).intValue()
        ));
    }

    private Double calculateCDD(ChainTx tx, List<ChainTxInput> inputs, Map<String, ChainTx> prevTxMap) {
        double cdd = 0.0;
        LocalDateTime currentTime = tx.getBlockTime();
        for (ChainTxInput input : inputs) {
            ChainTx prevTx = prevTxMap.get(input.getPrevTxHash());
            if (prevTx == null || prevTx.getBlockTime() == null) continue;
            long seconds = ChronoUnit.SECONDS.between(prevTx.getBlockTime(), currentTime);
            double days = seconds / (24.0 * 3600.0);
            if (days < 0) days = 0;
            double amount = input.getValue() == null ? 0.0 : input.getValue().doubleValue();
            cdd += amount * days;
        }
        return cdd;
    }
}