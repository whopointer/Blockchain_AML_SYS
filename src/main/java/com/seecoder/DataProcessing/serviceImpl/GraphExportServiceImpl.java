package com.seecoder.DataProcessing.serviceImpl;

import com.opencsv.CSVWriter;
import com.seecoder.DataProcessing.po.*;
import com.seecoder.DataProcessing.repository.*;
import com.seecoder.DataProcessing.service.GraphExportService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.annotation.PostConstruct;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
public class GraphExportServiceImpl implements GraphExportService {

    private static final int PAGE_SIZE = 500;

    @Autowired
    private ChainTxRepository chainTxRepository;
    @Autowired
    private ChainTxInputRepository chainTxInputRepository;
    @Autowired
    private ChainTxOutputRepository chainTxOutputRepository;

    // 黑白名单持久化 Repository
    @Autowired
    private BlacklistAddressRepository blacklistAddressRepository;
    @Autowired
    private WhitelistAddressRepository whitelistAddressRepository;

    @Value("${blacklist.file:src/data/sdn.csv}")
    private String blacklistFilePath;
    @Value("${whitelist.file:src/data/whitelist.csv}")
    private String whitelistFilePath;

    // 内存缓存，提高性能
    private Set<String> blacklist = new HashSet<>();
    private Set<String> whitelist = new HashSet<>();


//生成边表和class表，加载黑白名单
    @Override
    public ApiResponse<Void> refreshBlackWhitelist() {
        try {
            // 1. 从 CSV 文件加载
            Set<String> newBlacklist = loadAddressesFromCsv(blacklistFilePath);
            Set<String> newWhitelist = loadAddressesFromCsv(whitelistFilePath);

            // 2. 清空 MySQL 并重新插入
            blacklistAddressRepository.deleteAll();
            List<BlacklistAddress> blacklistEntities = newBlacklist.stream()
                    .map(addr -> {
                        BlacklistAddress entity = new BlacklistAddress();
                        entity.setAddress(addr);
                        return entity;
                    })
                    .collect(Collectors.toList());
            blacklistAddressRepository.saveAll(blacklistEntities);

            whitelistAddressRepository.deleteAll();
            List<WhitelistAddress> whitelistEntities = newWhitelist.stream()
                    .map(addr -> {
                        WhitelistAddress entity = new WhitelistAddress();
                        entity.setAddress(addr);
                        return entity;
                    })
                    .collect(Collectors.toList());
            whitelistAddressRepository.saveAll(whitelistEntities);

            // 3. 更新内存缓存
            blacklist = newBlacklist;
            whitelist = newWhitelist;

            log.info("黑白名单刷新成功，黑名单：{}，白名单：{}", blacklist.size(), whitelist.size());
            return ApiResponse.success(null, null);
        } catch (Exception e) {
            log.error("刷新黑白名单失败", e);
            return ApiResponse.error(500, "刷新失败: " + e.getMessage());
        }
    }

    private Set<String> loadAddressesFromCsv(String filePath) {
        Set<String> addresses = new HashSet<>();
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(filePath))) {
            String line;
            reader.readLine(); // 跳过表头
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 2) {
                    String addr = parts[1].trim();
                    if (!addr.isEmpty()) {
                        // 统一转小写，避免大小写不一致
                        addresses.add(addr.toLowerCase());
                    }
                }
            }
            log.info("从 {} 加载了 {} 个地址", filePath, addresses.size());
        } catch (IOException e) {
            log.warn("读取地址文件失败: {}，将使用空集合", filePath, e);
        }
        return addresses;
    }

    @Override
    @Transactional(readOnly = true)
    public ApiResponse<String> exportGraph(String chain,
                                           Long startHeight, Long endHeight,
                                           LocalDateTime startTime, LocalDateTime endTime,
                                           String edgesFilePath, String classesFilePath) {
        // 参数处理：优先使用时间范围
        boolean useTimeRange = (startTime != null && endTime != null);

        if (useTimeRange) {
            // 高度参数被忽略，使用时间范围
            log.info("使用时间范围导出：{} 到 {}", startTime, endTime);
        } else {
            if (startHeight == null) startHeight = 0L;
            if (endHeight == null) endHeight = Long.MAX_VALUE;
            log.info("使用高度范围导出：{} 到 {}", startHeight, endHeight);
        }

        // 生成默认文件路径
        if (edgesFilePath == null || edgesFilePath.trim().isEmpty()) {
            edgesFilePath = generateDefaultFilePath(chain, startHeight, endHeight, startTime, endTime, "edges");
        }
        if (classesFilePath == null || classesFilePath.trim().isEmpty()) {
            classesFilePath = generateDefaultFilePath(chain, startHeight, endHeight, startTime, endTime, "classes");
        }

        // 确保目录存在
        createParentDirs(edgesFilePath);
        createParentDirs(classesFilePath);

        long totalEdges = 0;
        Set<String> allAddresses = new HashSet<>();
        Set<String> edgeSet = new HashSet<>(); // 边去重

        int page = 0;
        try (CSVWriter edgesWriter = new CSVWriter(new FileWriter(edgesFilePath))) {
            edgesWriter.writeNext(new String[]{"from_address", "to_address"});

            while (true) {
                Pageable pageable = PageRequest.of(page, PAGE_SIZE,
                        Sort.by(Sort.Direction.ASC, "blockHeight", "txIndex"));
                Page<ChainTx> txPage;

                if (useTimeRange) {
                    txPage = chainTxRepository.findByChainAndBlockTimeBetween(
                            chain.toUpperCase(), startTime, endTime, pageable);
                } else {
                    txPage = chainTxRepository.findByChainAndBlockHeightBetween(
                            chain.toUpperCase(), startHeight, endHeight, pageable);
                }

                if (txPage.isEmpty()) break;

                List<ChainTx> txs = txPage.getContent();
                log.info("处理第 {} 批，共 {} 笔交易", page + 1, txs.size());

                // 预加载 inputs/outputs（仅比特币需要）
                List<Long> txIds = txs.stream().map(ChainTx::getId).collect(Collectors.toList());
                Map<Long, List<ChainTxInput>> inputsMap = null;
                Map<Long, List<ChainTxOutput>> outputsMap = null;
                if ("BTC".equalsIgnoreCase(chain)) {
                    List<ChainTxInput> allInputs = chainTxInputRepository.findByTransactionIdIn(txIds);
                    List<ChainTxOutput> allOutputs = chainTxOutputRepository.findByTransactionIdIn(txIds);
                    inputsMap = allInputs.stream()
                            .collect(Collectors.groupingBy(in -> in.getTransaction().getId()));
                    outputsMap = allOutputs.stream()
                            .collect(Collectors.groupingBy(out -> out.getTransaction().getId()));
                }

                for (ChainTx tx : txs) {
                    if ("ETH".equalsIgnoreCase(chain)) {
                        String from = tx.getFromAddress();
                        String to = tx.getToAddress();
                        if (from != null && to != null && !from.equals(to)) {
                            String edgeKey = from + "->" + to;
                            if (!edgeSet.contains(edgeKey)) {
                                edgesWriter.writeNext(new String[]{from, to});
                                edgeSet.add(edgeKey);
                                totalEdges++;
                            }
                            allAddresses.add(from);
                            allAddresses.add(to);
                        } else {
                            if (from != null) allAddresses.add(from);
                            if (to != null) allAddresses.add(to);
                        }
                    } else if ("BTC".equalsIgnoreCase(chain)) {
                        List<ChainTxInput> inputs = inputsMap.getOrDefault(tx.getId(), Collections.emptyList());
                        List<ChainTxOutput> outputs = outputsMap.getOrDefault(tx.getId(), Collections.emptyList());
                        Set<String> fromAddrs = inputs.stream()
                                .map(ChainTxInput::getAddress)
                                .filter(Objects::nonNull)
                                .collect(Collectors.toSet());
                        Set<String> toAddrs = outputs.stream()
                                .map(ChainTxOutput::getAddress)
                                .filter(Objects::nonNull)
                                .collect(Collectors.toSet());
                        allAddresses.addAll(fromAddrs);
                        allAddresses.addAll(toAddrs);
                        for (String from : fromAddrs) {
                            for (String to : toAddrs) {
                                if (!from.equals(to)) {
                                    String edgeKey = from + "->" + to;
                                    if (!edgeSet.contains(edgeKey)) {
                                        edgesWriter.writeNext(new String[]{from, to});
                                        edgeSet.add(edgeKey);
                                        totalEdges++;
                                    }
                                }
                            }
                        }
                    } else {
                        return ApiResponse.error(400, "不支持的链: " + chain);
                    }
                }
                page++;
            }
            log.info("边表导出完成，共 {} 条边（去重后），文件：{}", totalEdges, edgesFilePath);
        } catch (IOException e) {
            log.error("导出边表失败", e);
            return ApiResponse.error(500, "导出边表失败: " + e.getMessage());
        }

        // 生成 classes 表
        try (CSVWriter classesWriter = new CSVWriter(new FileWriter(classesFilePath))) {
            classesWriter.writeNext(new String[]{"address", "label"});
            for (String addr : allAddresses) {
                String lowerAddr = addr.toLowerCase(); // 统一小写匹配
                String label;
                if (blacklist.contains(lowerAddr)) {
                    label = "1";
                } else if (whitelist.contains(lowerAddr)) {
                    label = "2";
                } else {
                    label = "unknown";
                }
                classesWriter.writeNext(new String[]{addr, label});
            }
            log.info("Classes表导出完成，共 {} 条地址记录，文件：{}", allAddresses.size(), classesFilePath);
        } catch (IOException e) {
            log.error("导出classes表失败", e);
            return ApiResponse.error(500, "导出classes表失败: " + e.getMessage());
        }

        String result = String.format("导出完成。边表：%s（%d条边），classes表：%s（%d个地址）",
                edgesFilePath, totalEdges, classesFilePath, allAddresses.size());
        return ApiResponse.success(result, null);
    }

    private void createParentDirs(String filePath) {
        File file = new File(filePath);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
    }

    private String generateDefaultFilePath(String chain, Long startHeight, Long endHeight,
                                           LocalDateTime startTime, LocalDateTime endTime,
                                           String type) {
        String baseDir = System.getProperty("user.dir") + File.separator + "src" + File.separator + "data" + File.separator;
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String rangePart;
        if (startTime != null && endTime != null) {
            rangePart = startTime.toLocalDate().toString() + "_to_" + endTime.toLocalDate().toString();
        } else {
            rangePart = startHeight + "_" + endHeight;
        }
        return baseDir + chain + "_" + type + "_" + rangePart + "_" + timestamp + ".csv";
    }
}