package com.seecoder.DataProcessing.serviceImpl;

import com.opencsv.CSVWriter;
import com.seecoder.DataProcessing.po.*;
import com.seecoder.DataProcessing.repository.*;
import com.seecoder.DataProcessing.service.FeatureExportService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphAddressService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphTransactionService;
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
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service("ethereumFeatureExportService")
public class EthereumFeatureExportServiceImpl implements FeatureExportService {

    private static final String CHAIN_ETH = "ETH";
    private static final int PAGE_SIZE = 500;

    @Autowired
    private ChainTxRepository chainTxRepository;

    @Autowired
    private ChainTxInputRepository chainTxInputRepository;

    @Autowired
    private ChainTxOutputRepository chainTxOutputRepository;

    @Autowired
    private GraphAddressService graphAddressService;

    @Autowired
    private GraphTransactionService graphTransactionService; // 如需使用可保留

    @Override
    @Transactional(readOnly = true)
    public ApiResponse<String> exportFeatures(String chain,
                                              Long startHeight, Long endHeight,
                                              LocalDateTime startTime, LocalDateTime endTime,
                                              String filePath) {
        if (!CHAIN_ETH.equals(chain)) {
            return ApiResponse.error(400, "不支持的链: " + chain);
        }
        // 参数处理：优先使用时间范围，否则用高度范围
        if (startTime == null && endTime == null) {
            if (startHeight == null) startHeight = 0L;
            if (endHeight == null) endHeight = Long.MAX_VALUE;
        } else {
            if (startTime == null) startTime = LocalDateTime.of(2015, 7, 30, 0, 0);
            if (endTime == null) endTime = LocalDateTime.now();
        }

        // 生成默认文件路径（如果未提供）
        if (filePath == null || filePath.trim().isEmpty()) {
            String baseDir = System.getProperty("user.dir") + File.separator + "src" + File.separator + "data" + File.separator;
            File dir = new File(baseDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String rangePart;
            if (startTime != null && endTime != null) {
                rangePart = startTime.toLocalDate().toString() + "_to_" + endTime.toLocalDate().toString();
            } else {
                rangePart = startHeight + "_" + endHeight;
            }
            filePath = baseDir + chain + "_" + rangePart + "_" + timestamp + ".csv";
        }

        // 确保目标文件的父目录存在
        File targetFile = new File(filePath);
        File parentDir = targetFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        // 预计算全局地址复用次数
        Map<String, Integer> globalAddressCount = preComputeAddressCount();

        long totalTxs = 0;
        int page = 0;

        try (CSVWriter writer = new CSVWriter(new FileWriter(targetFile))) {
            writer.writeNext(TransactionFeatures.getCsvHeader());

            while (true) {
                Pageable pageable = PageRequest.of(page, PAGE_SIZE,
                        Sort.by(Sort.Direction.ASC, "blockHeight", "txIndex"));
                Page<ChainTx> txPage;
                if (startTime != null && endTime != null) {
                    txPage = chainTxRepository.findByChainAndBlockTimeBetween(
                            CHAIN_ETH, startTime, endTime, pageable);
                } else {
                    txPage = chainTxRepository.findByChainAndBlockHeightBetween(
                            CHAIN_ETH, startHeight, endHeight, pageable);
                }
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

                // 收集本批次所有发送方地址，用于批量查询邻居特征
                Set<String> fromAddresses = txs.stream()
                        .map(ChainTx::getFromAddress)
                        .filter(Objects::nonNull)
                        .collect(Collectors.toSet());

                // ========== 修改点：使用创世时间到 endTime 查询上游交易 ==========
                LocalDateTime genesis = LocalDateTime.of(2015, 7, 30, 0, 0);
                Map<String, GraphAddressService.NeighborStats> neighborStatsMap = graphAddressService.batchGetNeighborStats(
                        fromAddresses, endTime, genesis, endTime);
                // =============================================================

                for (ChainTx tx : txs) {
                    TransactionFeatures features = new TransactionFeatures();
                    List<ChainTxInput> inputs = inputsMap.getOrDefault(tx.getId(), Collections.emptyList());
                    List<ChainTxOutput> outputs = outputsMap.getOrDefault(tx.getId(), Collections.emptyList());

                    // ---------- 基础特征 ----------
                    features.setNumInputs(inputs.size());          // 通常为1
                    features.setNumOutputs(outputs.size());        // 通常为0或1
                    features.setTxFee(FeatureUtils.logTransform(tx.getFeeAsDouble()));
                    features.setTxSizeBytes(tx.getSizeBytes() != null ? tx.getSizeBytes() : 0L);
                    features.setFeeRate(calculateFeeRate(tx, features.getTxSizeBytes()));
                    features.setTotalInputAmt(FeatureUtils.logTransform(tx.getTotalInputAsDouble()));
                    features.setTotalOutputAmt(FeatureUtils.logTransform(tx.getTotalOutputAsDouble()));

                    features.setOutputStdDev(FeatureUtils.outputStdDev(outputs));

                    features.setLocktime(0L);                     // 以太坊无locktime
                    features.setIsCoinbase(0);                    // 以太坊无coinbase交易

                    // ---------- 邻居特征 ----------
                    GraphAddressService.NeighborStats stats = neighborStatsMap.get(tx.getFromAddress());
                    if (stats != null) {
                        features.setNeighborInAmtMean(stats.getAvgAmount());
                        features.setNeighborInAmtStd(stats.getStdAmount());
                        features.setNeighborInFeeMean(stats.getAvgFee());
                        features.setNeighborInDegreeMean(stats.getAvgInputs());
                        features.setNeighborOutDegreeMean(stats.getAvgOutputs());
                        features.setNeighborTimeSpan(stats.getAvgTimeSpan());
                    } else {
                        features.setNeighborInAmtMean(0.0);
                        features.setNeighborInAmtStd(0.0);
                        features.setNeighborInFeeMean(0.0);
                        features.setNeighborInDegreeMean(0.0);
                        features.setNeighborOutDegreeMean(0.0);
                        features.setNeighborTimeSpan(0.0);
                    }

                    // ---------- 其他特征 ----------
                    features.setCoindaysDestroyed(0.0);           // 以太坊无币龄概念

                    features.setRoundValueRatio(FeatureUtils.roundValueRatio(outputs));

                    // 地址复用次数：取发送方和接收方中最大的历史出现次数
                    int fromReuse = globalAddressCount.getOrDefault(tx.getFromAddress(), 0);
                    int toReuse = tx.getToAddress() != null ? globalAddressCount.getOrDefault(tx.getToAddress(), 0) : 0;
                    features.setAddressReuseCount(Math.max(fromReuse, toReuse));

                    // 循环标志
                    boolean loop = false;
                    if (tx.getToAddress() != null) {
                        loop = graphAddressService.existsCycle(
                                Collections.singletonList(tx.getFromAddress()),
                                Collections.singletonList(tx.getToAddress()),
                                5);
                    }
                    features.setLoopFlag(loop ? 1 : 0);

                    // 扇入扇出比
                    features.setFanInRatio((double) inputs.size() / (outputs.size() + 1));

                    writer.writeNext(features.toCsvRow());
                    totalTxs++;
                }
                page++;
            }
            log.info("以太坊特征导出完成，共 {} 笔交易", totalTxs);
            String resultMsg = "导出成功，共 " + totalTxs + " 笔交易，文件保存至: " + targetFile.getAbsolutePath();
            return ApiResponse.success(resultMsg, totalTxs);
        } catch (IOException e) {
            log.error("导出CSV失败", e);
            return ApiResponse.error(500, "导出失败: " + e.getMessage());
        }
    }

    /**
     * 预计算所有地址的出现次数（作为from或to）
     */
    private Map<String, Integer> preComputeAddressCount() {
        Map<String, Integer> countMap = new HashMap<>();
        // 查询 from 地址计数
        List<Object[]> fromCounts = chainTxRepository.countGroupByFromAddress(CHAIN_ETH);
        for (Object[] row : fromCounts) {
            String addr = (String) row[0];
            Number cnt = (Number) row[1];
            countMap.put(addr, cnt.intValue());
        }
        // 查询 to 地址计数
        List<Object[]> toCounts = chainTxRepository.countGroupByToAddress(CHAIN_ETH);
        for (Object[] row : toCounts) {
            String addr = (String) row[0];
            Number cnt = (Number) row[1];
            countMap.merge(addr, cnt.intValue(), Integer::sum);
        }
        return countMap;
    }

    private Double calculateFeeRate(ChainTx tx, Long sizeBytes) {
        if (sizeBytes == null || sizeBytes == 0) return 0.0;
        return tx.getFeeAsDouble() / sizeBytes;
    }
}