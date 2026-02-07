package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.session.Session;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

@Slf4j
@Service
public class GraphAddressService extends AbstractGraphService {

    /**
     * 缩短地址显示，格式为前5位+...+后5位
     * @param address 完整地址
     * @return 缩短后的地址
     */
    private String shortenAddress(String address) {
        if (address == null || address.length() <= 8) {
            return address;
        }
        return address.substring(0, 5) + "..." + address.substring(address.length() - 5);
    }
    
    /**
     * 格式化金额标签，根据链类型和金额大小选择不同的显示格式
     * @param amount 金额
     * @param chain 区块链类型
     * @return 格式化后的金额标签
     */
    private String formatAmountLabel(double amount, String chain) {
        String unit = (chain != null && chain.equalsIgnoreCase("ethereum")) ? "ETH" : "BNB";
        if (amount >= 1.0) {
            return String.format("%.3f %s", amount, unit);
        } else {
            return String.format("%.6f %s", amount, unit);
        }
    }
    
    /**
     * 解析时间戳并转换为UTC时间字符串
     * @param timestampStr 时间戳字符串，可以是数字格式或多种字符串格式
     * @return 格式化后的UTC时间字符串，解析失败返回原始字符串
     */
    private String parseAndFormatTimestamp(String timestampStr) {
        if (timestampStr == null || timestampStr.isEmpty()) {
            return timestampStr;
        }
        
        try {
            if (timestampStr.matches("-?\\d+(\\.\\d+)?([Ee][+-]?\\d+)?")) {
                double timestampDouble = Double.parseDouble(timestampStr);
                long milliseconds = (long) (timestampDouble * 1000);
                LocalDateTime utcDateTime = LocalDateTime.ofInstant(Instant.ofEpochMilli(milliseconds), ZoneId.of("UTC"));
                return utcDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            } else {
                LocalDateTime dateTime;
                
                String[] formats = {
                        "yyyy-MM-dd HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss.SSS",
                        "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
                };
                
                for (String format : formats) {
                    try {
                        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
                        dateTime = LocalDateTime.parse(timestampStr, formatter);
                        return dateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                    } catch (Exception e) {
                    }
                }
                
                dateTime = LocalDateTime.parse(timestampStr, DateTimeFormatter.ISO_DATE_TIME);
                return dateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            }
        } catch (Exception e) {
            log.warn("无法解析时间戳: {}", timestampStr);
            return timestampStr; // 解析失败时返回原始字符串
        }
    }

    /**
     * 查找两个地址之间的N跳交易路径
     * @param fromAddress 起始地址
     * @param toAddress 目标地址
     * @param maxHops 最大跳数，默认3，范围1-5
     * @return 包含路径节点、边、交易信息的响应
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findNhopTransactionPath(String fromAddress, String toAddress, Integer maxHops) {
        try {
            if (fromAddress == null || toAddress == null) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            if (maxHops == null || maxHops <= 0 || maxHops > 5) {
                maxHops = 3;
            }

            Session session = getSession();
            try {
                String pathQuery = "MATCH path = (a:Address {address: $fromAddress})" +
                        "-[:TRANSFER*1.." + maxHops + "]->(b:Address {address: $toAddress}) " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BNB')} ] AS nodeData, " +
                        "     [r IN relList | {tx_hash: coalesce(r.tx_hash, ''), amount: coalesce(toFloat(r.amount), 0.0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData LIMIT 50";

                Map<String, Object> params = new java.util.HashMap<>();
                params.put("fromAddress", fromAddress);
                params.put("toAddress", toAddress);

                Iterable<Map<String, Object>> queryResult;
                try {
                    queryResult = session.query(pathQuery, params);
                } catch (Exception e) {
                    log.error("执行Cypher查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                Map<String, Object> graphDic = new java.util.HashMap<>();
                List<Map<String, Object>> allNodeList = new java.util.ArrayList<>();
                List<Map<String, Object>> allEdgeList = new java.util.ArrayList<>();
                int totalTxCount = 0;
                String firstTime = null;
                String latestTime = null;

                List<List<Map<String, Object>>> allPaths = new ArrayList<>();
                List<List<Map<String, Object>>> allPathRels = new ArrayList<>();
                List<List<String>> allPathTimes = new ArrayList<>();
                
                for (Map<String, Object> result : queryResult) {
                    Object rawNodeDataObj = result.get("nodeData");
                    List<Map<String, Object>> nodeData = new ArrayList<>();
                    if (rawNodeDataObj instanceof List) {
                        nodeData = (List<Map<String, Object>>) rawNodeDataObj;
                    } else if (rawNodeDataObj != null) {
                        if (rawNodeDataObj.getClass().isArray()) {
                            Object[] array = (Object[]) rawNodeDataObj;
                            for (Object obj : array) {
                                if (obj instanceof Map) {
                                    nodeData.add((Map<String, Object>) obj);
                                }
                            }
                        } else if (rawNodeDataObj instanceof Map) {
                            nodeData.add((Map<String, Object>) rawNodeDataObj);
                        } else {
                            log.warn("Unexpected data type for nodeData: {}", rawNodeDataObj.getClass().getName());
                        }
                    }
                    
                    Object rawRelDataObj = result.get("relData");
                    List<Map<String, Object>> relData = new ArrayList<>();
                    if (rawRelDataObj instanceof List) {
                        relData = (List<Map<String, Object>>) rawRelDataObj;
                    } else if (rawRelDataObj != null) {
                        if (rawRelDataObj.getClass().isArray()) {
                            Object[] array = (Object[]) rawRelDataObj;
                            for (Object obj : array) {
                                if (obj instanceof Map) {
                                    relData.add((Map<String, Object>) obj);
                                }
                            }
                        } else if (rawRelDataObj instanceof Map) {
                            relData.add((Map<String, Object>) rawRelDataObj);
                        } else {
                            log.warn("Unexpected data type for relData: {}", rawRelDataObj.getClass().getName());
                        }
                    }
                    
                    List<String> txHashes = new ArrayList<>();
                    List<Double> amounts = new ArrayList<>();
                    List<String> times = new ArrayList<>();
                    
                    for (Map<String, Object> rel : relData) {
                        Object txHashObj = rel.get("tx_hash");
                        String txHash = txHashObj != null ? txHashObj.toString() : "";
                        txHashes.add(txHash);
                        
                        Object amountObj = rel.get("amount");
                        Double amount = 0.0;
                        if (amountObj != null) {
                            if (amountObj instanceof Double) {
                                amount = (Double) amountObj;
                            } else if (amountObj instanceof Number) {
                                amount = ((Number) amountObj).doubleValue();
                            } else {
                                try {
                                    amount = Double.parseDouble(amountObj.toString());
                                } catch (NumberFormatException e) {
                                    log.warn("Cannot parse amount: {}", amountObj);
                                }
                            }
                        }
                        amounts.add(amount);
                        
                        Object timeObj = rel.get("time");
                        String time = timeObj != null ? timeObj.toString() : "";
                        times.add(parseAndFormatTimestamp(time));
                    }
                    
                    allPaths.add(nodeData);
                    allPathRels.add(relData);
                    allPathTimes.add(times);
                    
                    totalTxCount += Math.min(Math.min(txHashes.size(), amounts.size()), times.size());
                    for (String t : times) {
                        if (t != null && !t.isEmpty()) {
                            if (firstTime == null || t.compareTo(firstTime) < 0) {
                                firstTime = t;
                            }
                            if (latestTime == null || t.compareTo(latestTime) > 0) {
                                latestTime = t;
                            }
                        }
                    }
                }
                
                Map<String, Integer> nodeLayers = calculateNodeLayers(allPaths, fromAddress, toAddress);
                
                Map<String, String> globalAddressToId = new java.util.HashMap<>();
                
                for (List<Map<String, Object>> nodeData : allPaths) {
                    for (int i = 0; i < nodeData.size(); i++) {
                        Map<String, Object> nodeMap = nodeData.get(i);
                        Object addressObj = nodeMap.get("address");
                        String address = addressObj != null ? addressObj.toString() : null;
                        
                        if (address == null) {
                            continue;
                        }
                        
                        Object riskLevelObj = nodeMap.get("risk_level");
                        Integer riskLevel = 0;
                        if (riskLevelObj != null) {
                            if (riskLevelObj instanceof Integer) {
                                riskLevel = (Integer) riskLevelObj;
                            } else if (riskLevelObj instanceof Number) {
                                riskLevel = ((Number) riskLevelObj).intValue();
                            } else {
                                try {
                                    riskLevel = Integer.parseInt(riskLevelObj.toString());
                                } catch (NumberFormatException e) {
                                    log.warn("Cannot parse risk level: {}", riskLevelObj);
                                }
                            }
                        }
                        
                        // 检查全局是否已有该地址的ID
                        String nodeId = globalAddressToId.get(address);
                        if (nodeId == null) {
                            // 生成新的ID并记录到全局映射
                            nodeId = java.util.UUID.randomUUID().toString();
                            globalAddressToId.put(address, nodeId);
                            
                            Map<String, Object> nodeItem = new java.util.HashMap<>();
                            nodeItem.put("id", nodeId);
                            nodeItem.put("label", shortenAddress(address));
                            nodeItem.put("title", address);
                            nodeItem.put("addr", address);
                            
                            // 使用calculateNodeLayers方法计算出的连续层级
                            int layer = nodeLayers.get(address);
                            nodeItem.put("layer", layer);
                            
                            if (riskLevel > 0) {
                                nodeItem.put("malicious", riskLevel);
                            }
                            
                            // 避免重复添加节点
                            boolean exists = false;
                            for (Map<String, Object> existingNode : allNodeList) {
                                if (existingNode.get("addr").equals(address)) {
                                    exists = true;
                                    break;
                                }
                            }
                            
                            if (!exists) {
                                allNodeList.add(nodeItem);
                            }
                        }
                    }
                }
                
                // 第四步：创建边列表
                for (int pathIdx = 0; pathIdx < allPaths.size(); pathIdx++) {
                    List<Map<String, Object>> nodeData = allPaths.get(pathIdx);
                    List<Map<String, Object>> relData = allPathRels.get(pathIdx);
                    List<String> times = allPathTimes.get(pathIdx);
                    
                    for (int i = 0; i < nodeData.size() - 1 && i < relData.size(); i++) {
                        if (i < nodeData.size() - 1) {
                            Map<String, Object> fromNodeData = nodeData.get(i);
                            Map<String, Object> toNodeData = nodeData.get(i + 1);
                            
                            String fromAddr = fromNodeData != null ? fromNodeData.get("address").toString() : "";
                            String toAddr = toNodeData != null ? toNodeData.get("address").toString() : "";
                            
                            // 检查边是否已存在
                            boolean edgeExists = allEdgeList.stream()
                                .anyMatch(edge -> edge.get("from").equals(fromAddr) && edge.get("to").equals(toAddr));
                            
                            if (!edgeExists) {
                                Map<String, Object> linkItem = new java.util.HashMap<>();
                                linkItem.put("from", fromAddr);
                                linkItem.put("to", toAddr);
                                
                                String chain = fromNodeData != null ? fromNodeData.get("chain").toString() : "BNB";
                                linkItem.put("label", formatAmountLabel(
                                    relData.get(i) != null ? 
                                        (Double) ((Map<String, Object>) relData.get(i)).get("amount") : 0.0, 
                                    chain));
                                linkItem.put("val", relData.get(i) != null ? 
                                    (Double) ((Map<String, Object>) relData.get(i)).get("amount") : 0.0);
                                linkItem.put("tx_time", i < times.size() ? times.get(i) : "");
                                
                                // 获取交易哈希
                                List<String> txHashList = new java.util.ArrayList<>();
                                if (relData.get(i) != null) {
                                    Object txHashObj = ((Map<String, Object>) relData.get(i)).get("tx_hash");
                                    String txHash = txHashObj != null ? txHashObj.toString() : "";
                                    if (!txHash.isEmpty()) {
                                        txHashList.add(txHash);
                                    }
                                }
                                linkItem.put("tx_hash_list", txHashList);
                                
                                allEdgeList.add(linkItem);
                            }
                        }
                    }
                }
                
                graphDic.put("node_list", allNodeList);
                graphDic.put("edge_list", allEdgeList);
                graphDic.put("tx_count", totalTxCount);
                graphDic.put("first_tx_datetime", firstTime != null ? firstTime : "");
                graphDic.put("latest_tx_datetime", latestTime != null ? latestTime : "");
                graphDic.put("address_first_tx_datetime", firstTime != null ? firstTime : "");
                graphDic.put("address_latest_tx_datetime", latestTime != null ? latestTime : "");

                Object nodeListObj = graphDic.get("node_list");
                long nodeListSize = 0;
                if (nodeListObj instanceof List) {
                    nodeListSize = ((List<?>) nodeListObj).size();
                }
                return ApiResponse.success(graphDic, nodeListSize);

            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("查找交易路径失败: {} -> {}", fromAddress, toAddress, e);
            return ApiResponse.error(500, "查找交易路径失败: " + e.getMessage());
        }
    }
    


    /**
     * 计算节点层级的方法，处理多节点多路径（包括循环路径）的情况
     * 保证起始节点层级为0，目标节点层级最大，其余节点合理分布
     */
    private Map<String, Integer> calculateNodeLayers(List<List<Map<String, Object>>> allPaths, String fromAddress, String toAddress) {
        Map<String, Integer> nodeLayers = new HashMap<>();

        // 如果没有路径，只设置起始节点为第0层
        if (allPaths == null || allPaths.isEmpty()) {
            nodeLayers.put(fromAddress, 0);
            return nodeLayers;
        }

        // 1. 初始化图结构，建立邻接关系，并收集所有节点
        Map<String, Set<String>> forwardEdges = new HashMap<>(); // 前向边（正常流向）
        Set<String> allNodes = new HashSet<>(); // 收集所有路径中的所有节点

        for (List<Map<String, Object>> path : allPaths) {
            for (int i = 0; i < path.size(); i++) {
                String nodeAddr = getNodeAddress(path.get(i));
                if (nodeAddr != null) {
                    allNodes.add(nodeAddr);
                }
            }

            for (int i = 0; i < path.size() - 1; i++) {
                String fromNode = getNodeAddress(path.get(i));
                String toNode = getNodeAddress(path.get(i + 1));

                if (fromNode != null && toNode != null && !fromNode.equals(toNode)) {
                    forwardEdges.computeIfAbsent(fromNode, k -> new HashSet<>()).add(toNode);
                }
            }
        }

        // 2. 使用广度优先搜索(BFS)从起始节点计算最短跳数（层级），确保起始节点为0
        Map<String, Integer> distances = new HashMap<>();
        java.util.Queue<String> queue = new java.util.LinkedList<>();
        distances.put(fromAddress, 0);
        queue.offer(fromAddress);

        while (!queue.isEmpty()) {
            String cur = queue.poll();
            int curDist = distances.getOrDefault(cur, 0);
            Set<String> nexts = forwardEdges.get(cur);
            if (nexts == null) continue;
            for (String nxt : nexts) {
                int nd = curDist + 1;
                if (!distances.containsKey(nxt) || nd < distances.get(nxt)) {
                    distances.put(nxt, nd);
                    queue.offer(nxt);
                }
            }
        }

        // 3. 对于未被BFS到达的节点，设置为 maxDistance + 1（保持它们位于右侧但不影响起始节点）
        int maxDistance = distances.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        for (String n : allNodes) {
            if (!distances.containsKey(n)) {
                distances.put(n, maxDistance + 1);
            }
        }

        // 4. 确保起始节点层级基准为0（防御性处理）并确保只有起始节点为0
        int startLayer = distances.getOrDefault(fromAddress, 0);
        // 将所有层级减去 startLayer，使起始节点为0
        Map<String, Integer> shifted = new HashMap<>();
        for (Map.Entry<String, Integer> e : distances.entrySet()) {
            shifted.put(e.getKey(), e.getValue() - startLayer);
        }

        // 如果有其他节点也被减为0，则将它们升到1，保证起始节点唯一为0
        for (Map.Entry<String, Integer> e : new HashMap<>(shifted).entrySet()) {
            if (!e.getKey().equals(fromAddress) && e.getValue() <= 0) {
                shifted.put(e.getKey(), 1);
            }
        }

        // 5. 规范化为连续层级（0,1,2,...），并确保起始节点仍为0
        Map<String, Integer> normalized = normalizeLayers(shifted);

        // 如果因某些情况起始节点不为0，则整体平移使其为0
        int normStart = normalized.getOrDefault(fromAddress, 0);
        if (normStart != 0) {
            Map<String, Integer> adjusted = new HashMap<>();
            for (Map.Entry<String, Integer> e : normalized.entrySet()) {
                adjusted.put(e.getKey(), e.getValue() - normStart);
            }
            // 再次确保只有起始节点为0
            for (Map.Entry<String, Integer> e : new HashMap<>(adjusted).entrySet()) {
                if (!e.getKey().equals(fromAddress) && e.getValue() <= 0) {
                    adjusted.put(e.getKey(), 1);
                }
            }
            normalized = normalizeLayers(adjusted);
        }

        // 6. 确保目标节点为最大层级（保持原有行为）
        int maxLayer = normalized.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        if (normalized.containsKey(toAddress)) {
            long countOfMaxLayerNodes = normalized.values().stream()
                .filter(layer -> layer.equals(maxLayer)).count();

            if (normalized.get(toAddress) < maxLayer || 
                (normalized.get(toAddress).equals(maxLayer) && countOfMaxLayerNodes > 1)) {
                normalized.put(toAddress, maxLayer + 1);
                normalized = reassignLayersWithTargetAtMax(normalized, toAddress);
                // 再次保证起始节点为0
                if (normalized.containsKey(fromAddress) && normalized.get(fromAddress) != 0) {
                    int shift = normalized.get(fromAddress);
                    Map<String, Integer> tmp = new HashMap<>();
                    for (Map.Entry<String, Integer> e : normalized.entrySet()) {
                        tmp.put(e.getKey(), e.getValue() - shift);
                    }
                    normalized = normalizeLayers(tmp);
                }
            }
        } else {
            normalized.put(toAddress, maxLayer + 1);
            normalized = reassignLayersWithTargetAtMax(normalized, toAddress);
            if (normalized.containsKey(fromAddress) && normalized.get(fromAddress) != 0) {
                int shift = normalized.get(fromAddress);
                Map<String, Integer> tmp = new HashMap<>();
                for (Map.Entry<String, Integer> e : normalized.entrySet()) {
                    tmp.put(e.getKey(), e.getValue() - shift);
                }
                normalized = normalizeLayers(tmp);
            }
        }

        return normalized;
    }
    
    /**
     * 重新分配层级，确保目标节点位于最大层级且层级连续
     */
    private Map<String, Integer> reassignLayersWithTargetAtMax(Map<String, Integer> nodeLayers, String targetAddress) {
        // 分离目标节点和其他节点的层级
        Integer targetLayer = nodeLayers.get(targetAddress);
        if (targetLayer == null) {
            return nodeLayers; // 目标节点不存在，直接返回
        }
        
        // 获取除目标节点外的所有其他层级
        List<Integer> otherLayers = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : nodeLayers.entrySet()) {
            if (!entry.getKey().equals(targetAddress)) {
                otherLayers.add(entry.getValue());
            }
        }
        
        // 对其他层级进行排序并去重，创建连续映射
        java.util.SortedSet<Integer> uniqueOtherLayers = new java.util.TreeSet<>(otherLayers);
        Map<Integer, Integer> otherLayerMapping = new java.util.HashMap<>();
        int layerIndex = 0;
        for (Integer layer : uniqueOtherLayers) {
            otherLayerMapping.put(layer, layerIndex++);
        }
        
        // 创建新的层级映射
        Map<String, Integer> result = new java.util.HashMap<>();
        for (Map.Entry<String, Integer> entry : nodeLayers.entrySet()) {
            if (entry.getKey().equals(targetAddress)) {
                // 目标节点设置为最大层级（其他层级数量）
                result.put(entry.getKey(), uniqueOtherLayers.size());
            } else {
                // 其他节点使用映射后的层级
                result.put(entry.getKey(), otherLayerMapping.get(entry.getValue()));
            }
        }
        
        return result;
    }
    
    /**
     * 将层级标准化为连续的整数序列（0, 1, 2, ...）
     */
    private Map<String, Integer> normalizeLayers(Map<String, Integer> nodeLayers) {
        if (nodeLayers.isEmpty()) {
            return nodeLayers;
        }
        
        // 获取所有唯一层级值并排序
        java.util.SortedSet<Integer> uniqueLayers = new java.util.TreeSet<>(nodeLayers.values());
        Map<Integer, Integer> layerMapping = new java.util.HashMap<>();
        
        // 创建从旧层级到新连续层级的映射
        int newLayerValue = 0;
        for (Integer oldLayer : uniqueLayers) {
            layerMapping.put(oldLayer, newLayerValue++);
        }
        
        // 创建新映射
        Map<String, Integer> normalizedLayers = new java.util.HashMap<>();
        for (Map.Entry<String, Integer> entry : nodeLayers.entrySet()) {
            normalizedLayers.put(entry.getKey(), layerMapping.get(entry.getValue()));
        }
        
        return normalizedLayers;
    }
    
    /**
     * 查找指定地址N跳内的所有关联地址
     * @param address 中心地址
     * @param maxHops 最大跳数，默认3，范围1-6
     * @return 关联地址列表及其基本信息
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<List<Map<String, Object>>> findAddressesWithinNHops(String address, Integer maxHops) {
        try {
            if (address == null || address.isEmpty()) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            if (maxHops == null || maxHops <= 0 || maxHops > 6) {
                maxHops = 3;
            }

            Session session = getSession();
            try {
                // 修复：将maxHops直接拼接到查询字符串中，而不是作为参数
                String query = "MATCH (a:Address {address: $address})" +
                        "-[r:TRANSFER*1.." + maxHops + "]-(b:Address) " +
                        "WHERE a <> b " +
                        "RETURN DISTINCT b.address as address, " +
                        "       COALESCE(b.chain, '') as chain, " +
                        "       COALESCE(b.first_seen, '') as firstSeen, " +
                        "       COALESCE(b.last_seen, '') as lastSeen, " +
                        "       COALESCE(b.risk_level, 0) as riskLevel, " +
                        "       COALESCE(b.tag, '') as tag " +
                        "LIMIT 50";

                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                Iterable<Map<String, Object>> queryResult;
                try {
                    queryResult = session.query(query, params);
                } catch (Exception e) {
                    log.error("执行N跳查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                List<Map<String, Object>> results = new ArrayList<>();
                for (Map<String, Object> record : queryResult) {
                    results.add(record);
                }

                return ApiResponse.success(results, (long) results.size());
            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("查找N跳内地址失败: {}", address, e);
            return ApiResponse.error(500, "查找地址失败: " + e.getMessage());
        }
    }

    /**
     * 获取地址转账统计信息
     * @param address 地址
     * @return 包含地址基本信息、转账统计和关联地址的响应
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> getAddressTransferStats(String address) {
        try {
            Map<String, Object> stats = new HashMap<>();

            Session session = getSession();
            try {
                // 1. 使用Cypher查询地址基本信息，避免使用Repository
                String addressQuery = "MATCH (a:Address {address: $address}) " +
                        "RETURN a.address as address, a.chain as chain, " +
                        "COALESCE(a.first_seen, '') as firstSeen, COALESCE(a.last_seen, '') as lastSeen, " +
                        "COALESCE(a.risk_level, 0) as riskLevel, COALESCE(a.tag, '') as tag, " +
                        "COALESCE(a.balance, 0) as balance, COALESCE(a.txCount, 0) as txCount";

                Map<String, Object> params = Collections.singletonMap("address", address);
                Iterable<Map<String, Object>> addressResult;
                try {
                    addressResult = session.query(addressQuery, params);
                } catch (Exception e) {
                    log.error("执行地址信息查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                for (Map<String, Object> record : addressResult) {
                    stats.putAll(record);
                }

                // 2. 获取转账统计（使用toFloat转换金额）
                String sentQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN SUM(toFloat(r.amount)) as totalSent";

                Iterable<Map<String, Object>> sentResult;
                try {
                    sentResult = session.query(sentQuery, params);
                } catch (Exception e) {
                    log.error("执行发送金额统计查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }
                
                Double totalSent = 0.0;
                for (Map<String, Object> record : sentResult) {
                    if (record.get("totalSent") != null) {
                        totalSent = ((Number) record.get("totalSent")).doubleValue();
                    }
                }

                String receivedQuery = "MATCH (a:Address {address: $address})<-[r:TRANSFER]-(b:Address) " +
                        "RETURN SUM(toFloat(r.amount)) as totalReceived";

                Iterable<Map<String, Object>> receivedResult;
                try {
                    receivedResult = session.query(receivedQuery, params);
                } catch (Exception e) {
                    log.error("执行接收金额统计查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }
                
                Double totalReceived = 0.0;
                for (Map<String, Object> record : receivedResult) {
                    if (record.get("totalReceived") != null) {
                        totalReceived = ((Number) record.get("totalReceived")).doubleValue();
                    }
                }

                stats.put("totalSent", totalSent);
                stats.put("totalReceived", totalReceived);

                // 3. 获取直接关联地址
                String toQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN DISTINCT b.address as toAddress";
                Iterable<Map<String, Object>> toResult = session.query(toQuery, params, false);

                List<String> transferTo = new ArrayList<>();
                for (Map<String, Object> record : toResult) {
                    transferTo.add((String) record.get("toAddress"));
                }

                String fromQuery = "MATCH (a:Address {address: $address})<-[r:TRANSFER]-(b:Address) " +
                        "RETURN DISTINCT b.address as fromAddress";
                Iterable<Map<String, Object>> fromResult = session.query(fromQuery, params, false);

                List<String> transferFrom = new ArrayList<>();
                for (Map<String, Object> record : fromResult) {
                    transferFrom.add((String) record.get("fromAddress"));
                }

                stats.put("transferToCount", transferTo.size());
                stats.put("transferFromCount", transferFrom.size());
                stats.put("transferToAddresses", transferTo);
                stats.put("transferFromAddresses", transferFrom);

            } finally {
                closeSession(session);
            }

            return ApiResponse.success(stats, null);
        } catch (Exception e) {
            log.error("获取地址转账统计失败: {}", address, e);
            return ApiResponse.error(500, "获取地址转账统计失败: " + e.getMessage());
        }
    }

    /**
     * 获取地址的直接连接（入度和出度地址）
     * @param address 地址
     * @return 包含出度和入度连接的响应
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> getDirectConnections(String address) {
        try {
            Map<String, Object> connections = new HashMap<>();

            Session session = getSession();
            try {
                // 获取出度连接
                String outgoingQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN DISTINCT b.address as address";
                Map<String, Object> params = Collections.singletonMap("address", address);

                Iterable<Map<String, Object>> outgoingResult;
                try {
                    outgoingResult = session.query(outgoingQuery, params);
                } catch (Exception e) {
                    log.error("执行出度连接查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }
                
                List<String> outgoingConnections = new ArrayList<>();
                for (Map<String, Object> record : outgoingResult) {
                    outgoingConnections.add((String) record.get("address"));
                }

                // 获取入度连接
                String incomingQuery = "MATCH (a:Address)<-[r:TRANSFER]-(b:Address {address: $address}) " +
                        "RETURN DISTINCT a.address as address";
                
                Iterable<Map<String, Object>> incomingResult;
                try {
                    incomingResult = session.query(incomingQuery, params);
                } catch (Exception e) {
                    log.error("执行入度连接查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }
                
                List<String> incomingConnections = new ArrayList<>();
                for (Map<String, Object> record : incomingResult) {
                    incomingConnections.add((String) record.get("address"));
                }

                connections.put("outgoingConnections", outgoingConnections);
                connections.put("incomingConnections", incomingConnections);
                connections.put("outgoingCount", outgoingConnections.size());
                connections.put("incomingCount", incomingConnections.size());
                connections.put("totalConnections", outgoingConnections.size() + incomingConnections.size());

            } finally {
                closeSession(session);
            }

            return ApiResponse.success(connections, null);
        } catch (Exception e) {
            log.error("获取直接连接失败: {}", address, e);
            return ApiResponse.error(500, "获取直接连接失败: " + e.getMessage());
        }
    }

    /**
     * 更新地址的风险等级
     * @param address 地址
     * @param riskLevel 风险等级
     * @return 更新结果响应
     */
    @Transactional(transactionManager = "neo4jTransactionManager")
    public ApiResponse<Void> updateAddressRiskLevel(String address, Integer riskLevel) {
        try {
            Session session = getSession();
            try {
                String checkQuery = "MATCH (a:Address {address: $address}) RETURN a.address as address";
                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                Iterable<Map<String, Object>> checkResult;
                try {
                    checkResult = session.query(checkQuery, params);
                } catch (Exception e) {
                    log.error("执行地址存在性检查查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                boolean exists = false;
                for (Map<String, Object> record : checkResult) {
                    if (record.get("address") != null) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
                    return ApiResponse.error(404, "地址不存在: " + address);
                }

                String updateQuery = "MATCH (a:Address {address: $address}) SET a.risk_level = $riskLevel";
                params.put("riskLevel", riskLevel);

                try {
                    session.query(updateQuery, params);
                } catch (Exception e) {
                    log.error("执行更新风险等级查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "更新失败: " + e.getMessage());
                }

                log.info("更新地址风险等级: {} -> {}", address, riskLevel);
                return ApiResponse.success(null, null);
            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("更新地址风险等级失败: {}", address, e);
            return ApiResponse.error(500, "更新地址风险等级失败: " + e.getMessage());
        }
    }

    /**
     * 更新地址的标签
     * @param address 地址
     * @param tag 标签内容
     * @return 更新结果响应
     */
    @Transactional(transactionManager = "neo4jTransactionManager")
    public ApiResponse<Void> tagAddress(String address, String tag) {
        try {
            Session session = getSession();
            try {
                String checkQuery = "MATCH (a:Address {address: $address}) RETURN a.address as address";
                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                Iterable<Map<String, Object>> checkResult;
                try {
                    checkResult = session.query(checkQuery, params);
                } catch (Exception e) {
                    log.error("执行地址存在性检查查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                boolean exists = false;
                for (Map<String, Object> record : checkResult) {
                    if (record.get("address") != null) {
                        exists = true;
                        break;
                    }
                }

                if (!exists) {
                    return ApiResponse.error(404, "地址不存在: " + address);
                }

                String updateQuery = "MATCH (a:Address {address: $address}) SET a.tag = $tag";
                params.put("tag", tag);

                try {
                    session.query(updateQuery, params);
                } catch (Exception e) {
                    log.error("执行更新标签查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "更新失败: " + e.getMessage());
                }

                log.info("更新地址标签: {} -> {}", address, tag);
                return ApiResponse.success(null, null);
            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("更新地址标签失败: {}", address, e);
            return ApiResponse.error(500, "更新地址标签失败: " + e.getMessage());
        }
    }
    
    /**
     * 辅助方法：从节点数据中提取地址
     */
    private String getNodeAddress(Map<String, Object> nodeData) {
        if (nodeData == null) {
            return null;
        }
        Object addrObj = nodeData.get("address");
        return addrObj != null ? addrObj.toString() : null;
    }
}