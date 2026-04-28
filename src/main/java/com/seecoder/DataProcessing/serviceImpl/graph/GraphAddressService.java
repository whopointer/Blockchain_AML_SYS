package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.util.GraphFormatUtils;
import com.seecoder.DataProcessing.util.GraphLayerCalculator;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.session.Session;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;

@Slf4j
@Service
public class GraphAddressService extends AbstractGraphService {

    private static final int DEFAULT_MAX_HOPS = 3;
    private static final int MAX_ALLOWED_HOPS = 6;
    private static final int DEFAULT_PATH_LIMIT = 50;
    private static final int MAX_PATH_LIMIT = 100;
    private static final int DEFAULT_NHOPS_LIMIT = 100;
    private static final int MAX_NHOPS_LIMIT = 200;
    private static final int QUERY_TIMEOUT_SECONDS = 30;
    
    // 按跳数动态调整结果限制：跳数越大，可返回路径数应越少
    private static final Map<Integer, Integer> HOPS_TO_PATH_LIMIT;
    private static final Map<Integer, Integer> HOPS_TO_NHOPS_LIMIT;
    
    static {
        Map<Integer, Integer> pathMap = new HashMap<>();
        pathMap.put(1, 100);   // 1跳：最多100条路径
        pathMap.put(2, 80);    // 2跳：最多80条路径
        pathMap.put(3, 50);    // 3跳：最多50条路径
        pathMap.put(4, 30);    // 4跳：最多30条路径
        pathMap.put(5, 20);    // 5跳：最多20条路径
        pathMap.put(6, 10);    // 6跳：最多10条路径
        HOPS_TO_PATH_LIMIT = Collections.unmodifiableMap(pathMap);
        
        Map<Integer, Integer> nhopsMap = new HashMap<>();
        nhopsMap.put(1, 200);
        nhopsMap.put(2, 150);
        nhopsMap.put(3, 100);
        nhopsMap.put(4, 50);
        nhopsMap.put(5, 30);
        nhopsMap.put(6, 20);
        HOPS_TO_NHOPS_LIMIT = Collections.unmodifiableMap(nhopsMap);
    }

    private int getConfiguredMaxHops(Integer maxHops) {
        if (maxHops == null || maxHops <= 0) {
            return DEFAULT_MAX_HOPS;
        }
        // 限制范围为 [1, MAX_ALLOWED_HOPS]
        int clamped = Math.max(1, Math.min(maxHops, MAX_ALLOWED_HOPS));
        
        // 记录实际使用的跳数，便于问题排查
        if (maxHops != clamped) {
            log.info("跳数参数 {} 被限制为 {} (最大允许: {})", maxHops, clamped, MAX_ALLOWED_HOPS);
        }
        return clamped;
    }
    
    /**
     * 根据跳数获取动态调整的路径限制
     * 跳数越大，路径可能呈指数增长，因此需要更严格的限制
     * @param maxHops 查询的跳数
     * @return 建议的结果限制
     */
    private int getDynamicPathLimit(Integer maxHops) {
        int hops = getConfiguredMaxHops(maxHops);
        Integer dynamicLimit = HOPS_TO_PATH_LIMIT.get(hops);
        return dynamicLimit != null ? dynamicLimit : DEFAULT_PATH_LIMIT;
    }
    
    /**
     * 根据跳数获取动态调整的N-hop限制
     */
    private int getDynamicNHopsLimit(Integer maxHops) {
        int hops = getConfiguredMaxHops(maxHops);
        Integer dynamicLimit = HOPS_TO_NHOPS_LIMIT.get(hops);
        return dynamicLimit != null ? dynamicLimit : DEFAULT_NHOPS_LIMIT;
    }

    private int getConfiguredPathLimit(Integer limit) {
        if (limit == null || limit <= 0) {
            return DEFAULT_PATH_LIMIT;
        }
        return Math.min(limit, MAX_PATH_LIMIT);
    }

    private int getConfiguredNHopsLimit(Integer limit) {
        if (limit == null || limit <= 0) {
            return DEFAULT_NHOPS_LIMIT;
        }
        return Math.min(limit, MAX_NHOPS_LIMIT);
    }

    /**
     * 查找两个地址之间的交易路径
     * @param fromAddress 起始地址
     * @param toAddress 目标地址
     * @param maxHops 最大跳数，范围1-6，默认为3
     * @param limit 返回路径数量限制，默认为50，最大100
     * @return 包含路径节点、边、交易信息的响应
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findNhopTransactionPath(String fromAddress, String toAddress, Integer maxHops, Integer limit) {
        try {
            if (fromAddress == null || toAddress == null) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            int effectiveMaxHops = getConfiguredMaxHops(maxHops);
            // 优先使用用户指定的 limit，否则使用动态限制
            int effectiveLimit = (limit != null && limit > 0) 
                ? getConfiguredPathLimit(limit) 
                : getDynamicPathLimit(maxHops);

            Session session = getSession();
            try {
                String pathQuery = "MATCH path = (a:Address {address: $fromAddress})" +
                        "-[:TRANSFER*1.." + effectiveMaxHops + "]->(b:Address {address: $toAddress}) " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BTC')} ] AS nodeData, " +
                        "     [r IN relList | {txHash: coalesce(r.txHash, ''), amount: coalesce(toFloat(r.amount), 0.0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData LIMIT " + effectiveLimit;

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
                        Object txHashObj = rel.get("txHash");
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
                        times.add(GraphFormatUtils.parseAndFormatTimestamp(time));
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
                
                Map<String, Integer> nodeLayers = GraphLayerCalculator.calculateNodeLayers(allPaths, fromAddress, toAddress);
                
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
                        
                        // 直接使用address作为节点id
                        String nodeId = address;
                        
                        Map<String, Object> nodeItem = new java.util.HashMap<>();
                        nodeItem.put("id", nodeId);
                        nodeItem.put("label", GraphFormatUtils.shortenAddress(address));
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
                                
                                String chain = fromNodeData != null ? fromNodeData.get("chain").toString() : "BTC";
                                linkItem.put("label", GraphFormatUtils.formatAmountLabel(
                                    relData.get(i) != null ? 
                                        (Double) ((Map<String, Object>) relData.get(i)).get("amount") : 0.0, 
                                    chain));
                                linkItem.put("val", relData.get(i) != null ? 
                                    (Double) ((Map<String, Object>) relData.get(i)).get("amount") : 0.0);
                                linkItem.put("tx_time", i < times.size() ? times.get(i) : "");
                                
                                // 获取交易哈希
                                List<String> txHashList = new java.util.ArrayList<>();
                                if (relData.get(i) != null) {
                                    Object txHashObj = ((Map<String, Object>) relData.get(i)).get("txHash");
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
     * 查找指定地址N跳内的所有关联地址
     * @param address 中心地址
     * @param maxHops 最大跳数，范围1-6，默认为3
     * @param limit 返回路径数量限制，默认为100，最大200
     * @return 包含节点列表、边列表、交易计数等信息的响应
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findAddressesWithinNHops(String address, Integer maxHops, Integer limit) {
        try {
            if (address == null || address.isEmpty()) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            int effectiveMaxHops = getConfiguredMaxHops(maxHops);
            // 优先使用用户指定的 limit，否则使用动态限制
            int effectiveLimit = (limit != null && limit > 0) 
                ? getConfiguredNHopsLimit(limit) 
                : getDynamicNHopsLimit(maxHops);

            Session session = getSession();
            try {
                String incomeQuery = "MATCH path = (start:Address {address: $address})<-[:TRANSFER*1.." + effectiveMaxHops + "]-(income:Address) " +
                        "WHERE start <> income " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BTC'), first_seen: coalesce(n.first_seen, ''), last_seen: coalesce(n.last_seen, '')} ] AS nodeData, " +
                        "     [r IN relList | {txHash: coalesce(r.txHash, ''), amount: coalesce(toFloat(r.amount), 0.0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData, 'income' as direction LIMIT " + effectiveLimit;

                String outcomeQuery = "MATCH path = (start:Address {address: $address})-[:TRANSFER*1.." + effectiveMaxHops + "]->(outcome:Address) " +
                        "WHERE start <> outcome " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BTC'), first_seen: coalesce(n.first_seen, ''), last_seen: coalesce(n.last_seen, '')} ] AS nodeData, " +
                        "     [r IN relList | {txHash: coalesce(r.txHash, ''), amount: coalesce(toFloat(r.amount), 0.0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData, 'outcome' as direction LIMIT " + effectiveLimit;

                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                // 查询收入侧
                Iterable<Map<String, Object>> incomeQueryResult;
                try {
                    incomeQueryResult = session.query(incomeQuery, params);
                } catch (Exception e) {
                    log.error("执行收入侧N跳查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                // 查询支出侧
                Iterable<Map<String, Object>> outcomeQueryResult;
                try {
                    outcomeQueryResult = session.query(outcomeQuery, params);
                } catch (Exception e) {
                    log.error("执行支出侧N跳查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                // 处理查询结果，构建节点和边列表
                Map<String, Object> graphDic = new HashMap<>();
                List<Map<String, Object>> allNodeList = new ArrayList<>();
                List<Map<String, Object>> allEdgeList = new ArrayList<>();
                int totalTxCount = 0;
                String firstTime = null;
                String latestTime = null;
                String addressFirstTime = null;
                String addressLatestTime = null;

                List<List<Map<String, Object>>> allPaths = new ArrayList<>();
                List<List<Map<String, Object>>> allPathRels = new ArrayList<>();
                List<List<String>> allPathTimes = new ArrayList<>();
                List<String> allPathDirections = new ArrayList<>();
                
                // 处理收入侧结果
                for (Map<String, Object> result : incomeQueryResult) {
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
                    
                    String direction = (String) result.get("direction");
                    
                    List<String> txHashes = new ArrayList<>();
                    List<Double> amounts = new ArrayList<>();
                    List<String> times = new ArrayList<>();
                    
                    for (Map<String, Object> rel : relData) {
                        Object txHashObj = rel.get("txHash");
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
                        times.add(GraphFormatUtils.parseAndFormatTimestamp(time));
                    }
                    
                    allPaths.add(nodeData);
                    allPathRels.add(relData);
                    allPathTimes.add(times);
                    allPathDirections.add(direction);
                    
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
                
                // 处理支出侧结果
                for (Map<String, Object> result : outcomeQueryResult) {
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
                    
                    String direction = (String) result.get("direction");
                    
                    List<String> txHashes = new ArrayList<>();
                    List<Double> amounts = new ArrayList<>();
                    List<String> times = new ArrayList<>();
                    
                    for (Map<String, Object> rel : relData) {
                        Object txHashObj = rel.get("txHash");
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
                        times.add(GraphFormatUtils.parseAndFormatTimestamp(time));
                    }
                    
                    allPaths.add(nodeData);
                    allPathRels.add(relData);
                    allPathTimes.add(times);
                    allPathDirections.add(direction);
                    
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
                
                // 计算节点层级，中心节点为0，收入侧为负值，支出侧为正值
                Map<String, Integer> nodeLayers = GraphLayerCalculator.calculateNodeLayersForNHops(allPaths, allPathDirections, address);
                
                Map<String, String> globalAddressToId = new HashMap<>();
                
                // 添加起始节点
                String startNodeId = address;
                Map<String, Object> startNodeItem = new HashMap<>();
                startNodeItem.put("id", startNodeId);
                startNodeItem.put("label", GraphFormatUtils.shortenAddress(address));
                startNodeItem.put("title", address);
                startNodeItem.put("addr", address);
                startNodeItem.put("layer", 0); // 起始节点layer为0
                globalAddressToId.put(address, startNodeId);
                allNodeList.add(startNodeItem);
                
                for (List<Map<String, Object>> nodeData : allPaths) {
                    for (int i = 0; i < nodeData.size(); i++) {
                        Map<String, Object> nodeMap = nodeData.get(i);
                        Object addressObj = nodeMap.get("address");
                        String nodeAddress = addressObj != null ? addressObj.toString() : null;
                        
                        if (nodeAddress == null || nodeAddress.equals(address)) {
                            continue; // 跳过起始节点，因为它已经被添加
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
                        
                        Object firstSeenObj = nodeMap.get("first_seen");
                        String firstSeen = firstSeenObj != null ? firstSeenObj.toString() : "";
                        if (firstSeen != null && !firstSeen.isEmpty() && (addressFirstTime == null || firstSeen.compareTo(addressFirstTime) < 0)) {
                            addressFirstTime = firstSeen;
                        }
                        
                        Object lastSeenObj = nodeMap.get("last_seen");
                        String lastSeen = lastSeenObj != null ? lastSeenObj.toString() : "";
                        if (lastSeen != null && !lastSeen.isEmpty() && (addressLatestTime == null || lastSeen.compareTo(addressLatestTime) > 0)) {
                            addressLatestTime = lastSeen;
                        }
                        
                        // 检查全局是否已有该地址的ID
                        String nodeId = globalAddressToId.get(nodeAddress);
                        if (nodeId == null) {
                            // 直接使用地址作为ID
                            nodeId = nodeAddress;
                            globalAddressToId.put(nodeAddress, nodeId);
                            
                            Map<String, Object> nodeItem = new HashMap<>();
                            nodeItem.put("id", nodeId);
                            nodeItem.put("label", GraphFormatUtils.shortenAddress(nodeAddress));
                            nodeItem.put("title", nodeAddress);
                            nodeItem.put("addr", nodeAddress);
                            
                            // 使用calculateBidirectionalNodeLayersWithDirection方法计算出的层级
                            int layer = nodeLayers.get(nodeAddress);
                            nodeItem.put("layer", layer);
                            
                            if (riskLevel > 0) {
                                nodeItem.put("malicious", riskLevel);
                            }
                            
                            // 避免重复添加节点
                            boolean exists = false;
                            for (Map<String, Object> existingNode : allNodeList) {
                                if (existingNode.get("addr").equals(nodeAddress)) {
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
                
                // 创建边列表
                for (int pathIdx = 0; pathIdx < allPaths.size(); pathIdx++) {
                    List<Map<String, Object>> nodeData = allPaths.get(pathIdx);
                    List<Map<String, Object>> relData = allPathRels.get(pathIdx);
                    List<String> times = allPathTimes.get(pathIdx);
                    String direction = allPathDirections.get(pathIdx);
                    
                    for (int i = 0; i < nodeData.size() - 1 && i < relData.size(); i++) {
                        if (i < nodeData.size() - 1) {
                            Map<String, Object> fromNodeData;
                            Map<String, Object> toNodeData;
                            String fromAddr;
                            String toAddr;
                            
                            // 根据路径方向调整from和to的顺序
                            if ("income".equals(direction)) {
                                // 收入侧路径，节点顺序与实际转账方向相反
                                fromNodeData = nodeData.get(i + 1);
                                toNodeData = nodeData.get(i);
                                fromAddr = fromNodeData != null ? fromNodeData.get("address").toString() : "";
                                toAddr = toNodeData != null ? toNodeData.get("address").toString() : "";
                            } else {
                                // 支出侧路径，节点顺序与实际转账方向一致
                                fromNodeData = nodeData.get(i);
                                toNodeData = nodeData.get(i + 1);
                                fromAddr = fromNodeData != null ? fromNodeData.get("address").toString() : "";
                                toAddr = toNodeData != null ? toNodeData.get("address").toString() : "";
                            }
                            
                            // 检查边是否已存在
                            boolean edgeExists = allEdgeList.stream()
                                .anyMatch(edge -> edge.get("from").equals(fromAddr) && edge.get("to").equals(toAddr));
                            
                            if (!edgeExists) {
                                Map<String, Object> linkItem = new HashMap<>();
                                linkItem.put("from", fromAddr);
                                linkItem.put("to", toAddr);
                                
                                String chain = fromNodeData != null ? fromNodeData.get("chain").toString() : "BTC";
                                linkItem.put("label", GraphFormatUtils.formatAmountLabel(
                                    relData.get(i) != null ? 
                                        (Double) ((Map<String, Object>) relData.get(i)).get("amount") : 0.0, 
                                    chain));
                                linkItem.put("val", relData.get(i) != null ? 
                                    (Double) ((Map<String, Object>) relData.get(i)).get("amount") : 0.0);
                                linkItem.put("tx_time", i < times.size() ? times.get(i) : "");
                                
                                // 获取交易哈希
                                List<String> txHashList = new ArrayList<>();
                                if (relData.get(i) != null) {
                                    Object txHashObj = ((Map<String, Object>) relData.get(i)).get("txHash");
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
                graphDic.put("first_tx_time", firstTime != null ? firstTime : "");
                graphDic.put("latest_tx_time", latestTime != null ? latestTime : "");
                graphDic.put("address_first_tx_time", addressFirstTime != null ? GraphFormatUtils.parseAndFormatTimestamp(addressFirstTime) : "");
                graphDic.put("address_latest_tx_time", addressLatestTime != null ? GraphFormatUtils.parseAndFormatTimestamp(addressLatestTime) : "");


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




    // ============ 新增 ============
    /**
     * 获取指定地址在时间范围内的直接交易对手地址
     */
    public Set<String> getNeighborAddresses(String address, LocalDateTime startTime, LocalDateTime endTime) {
        String cypher = "MATCH (a:Address {address: $addr}) -[:TRANSFER]-(neighbor:Address) " +
                "WHERE a <> neighbor " +
                "OPTIONAL MATCH (a)-[:TRANSFER]->(t:Transaction)-[:TRANSFER]->(neighbor) " +
                "WHERE t.time >= $start AND t.time <= $end " +
                "WITH a, neighbor, COUNT(t) AS cnt " +
                "WHERE cnt > 0 " +
                "RETURN DISTINCT neighbor.address";
        Map<String, Object> params = new HashMap<>();
        params.put("addr", address);
        params.put("start", startTime.toString());
        params.put("end", endTime.toString());

        Session session = getSession();
        try {
            Iterable<Map<String, Object>> result = session.query(cypher, params);
            Set<String> neighbors = new HashSet<>();
            for (Map<String, Object> row : result) {
                neighbors.add((String) row.get("neighbor.address"));
            }
            return neighbors;
        } finally {
            closeSession(session);
        }
    }

    /**
     * 获取指定地址在时间范围内的交易哈希
     */
    public Set<String> getTransactionHashes(String address, LocalDateTime startTime, LocalDateTime endTime) {
        // 只获取地址作为发送方的交易哈希（SPENT 关系）
        String cypher =
                "MATCH (a:Address {address: $addr})-[:SPENT]->(tx:Transaction) " +
                        "WHERE tx.time >= $start AND tx.time <= $end " +
                        "RETURN DISTINCT tx.txHash";

        Map<String, Object> params = new HashMap<>();
        params.put("addr", address);
        params.put("start", startTime.toString());
        params.put("end", endTime.toString());

        Session session = getSession();
        try {
            Iterable<Map<String, Object>> result = session.query(cypher, params);
            Set<String> txHashes = new HashSet<>();
            for (Map<String, Object> row : result) {
                txHashes.add((String) row.get("tx.txHash"));
            }
            return txHashes;
        } finally {
            closeSession(session);
        }
    }

    /**
     * 判断是否存在从输出地址到输入地址的循环路径（至少2跳）
     * @param inputAddrs  输入地址列表（发送方）
     * @param outputAddrs 输出地址列表（接收方）
     * @param maxHops     最大跳数，建议5-6，避免查询过重
     * @return true 表示存在循环
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public boolean existsCycle(List<String> inputAddrs, List<String> outputAddrs, int maxHops) {
        // 参数校验
        if (inputAddrs == null || outputAddrs == null ||
                inputAddrs.isEmpty() || outputAddrs.isEmpty()) {
            return false;
        }
        // 去重
        List<String> distinctInputs = new ArrayList<>(new HashSet<>(inputAddrs));
        List<String> distinctOutputs = new ArrayList<>(new HashSet<>(outputAddrs));

        Session session = getSession();
        try {
            // Cypher 查询：检查是否存在从 outputAddrs 中任一地址出发，
            // 经过 2 到 maxHops 跳后到达 inputAddrs 中任一地址的路径
            String cypher =
                    "MATCH path = (out:Address)-[*2.." + maxHops + "]->(in:Address) " +
                            "WHERE out.address IN $outputAddrs AND in.address IN $inputAddrs " +
                            "RETURN count(path) > 0 AS exists";
            Map<String, Object> params = new HashMap<>();
            params.put("outputAddrs", distinctOutputs);
            params.put("inputAddrs", distinctInputs);

            Iterable<Map<String, Object>> result = session.query(cypher, params);
            for (Map<String, Object> row : result) {
                Boolean exists = (Boolean) row.get("exists");
                return exists != null && exists;
            }
            return false;
        } catch (Exception e) {
            log.error("检测交易循环失败", e);
            return false; // 异常时保守返回 false
        } finally {
            closeSession(session);
        }
    }


    /**
     * 上游交易统计值内部类
     */
    public static class NeighborStats {
        private Double avgAmount = 0.0;
        private Double stdAmount = 0.0;
        private Double avgFee = 0.0;
        private Double avgInputs = 0.0;
        private Double avgOutputs = 0.0;
        private Double avgTimeSpan = 0.0;

        // getters and setters
        public Double getAvgAmount() { return avgAmount; }
        public void setAvgAmount(Double avgAmount) { this.avgAmount = avgAmount; }
        public Double getStdAmount() { return stdAmount; }
        public void setStdAmount(Double stdAmount) { this.stdAmount = stdAmount; }
        public Double getAvgFee() { return avgFee; }
        public void setAvgFee(Double avgFee) { this.avgFee = avgFee; }
        public Double getAvgInputs() { return avgInputs; }
        public void setAvgInputs(Double avgInputs) { this.avgInputs = avgInputs; }
        public Double getAvgOutputs() { return avgOutputs; }
        public void setAvgOutputs(Double avgOutputs) { this.avgOutputs = avgOutputs; }
        public Double getAvgTimeSpan() { return avgTimeSpan; }
        public void setAvgTimeSpan(Double avgTimeSpan) { this.avgTimeSpan = avgTimeSpan; }
    }
    /**
     * 批量获取多个地址的上游交易统计（作为接收方时的历史收入交易）
     * @param addresses  地址集合
     * @param currentTime 当前时间（用于计算时间间隔）
     * @param startTime   查询时间范围开始
     * @param endTime     查询时间范围结束
     * @return 地址 -> NeighborStats 的映射
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public Map<String, NeighborStats> batchGetNeighborStats(Set<String> addresses,
                                                            LocalDateTime currentTime,
                                                            LocalDateTime startTime,
                                                            LocalDateTime endTime) {
        if (addresses == null || addresses.isEmpty()) {
            return Collections.emptyMap();
        }

        Session session = getSession();
        try {
            // 使用 UNWIND 分别处理关系列表和交易列表，避免 avg([...]) 语法问题
            String cypher = "UNWIND $addresses AS addr " +
                    "MATCH (addrNode:Address {address: addr})<-[r:TRANSFER]-(tx:Transaction) " +
                    "WHERE tx.time >= $startTime AND tx.time <= $endTime " +
                    "WITH addr, COLLECT(r) AS rels, COLLECT(tx) AS txs " +
                    // 计算关系金额的平均值和标准差
                    "UNWIND rels AS rel " +
                    "WITH addr, txs, AVG(toFloat(rel.amount)) AS avgAmount, STDEV(toFloat(rel.amount)) AS stdAmount " +
                    // 计算交易相关特征
                    "UNWIND txs AS tx " +
                    "WITH addr, avgAmount, stdAmount, " +
                    "     AVG(toFloat(tx.fee)) AS avgFee, " +
                    "     AVG(COALESCE(tx.numInputs, 0)) AS avgInputs, " +      // 如果节点无此属性，默认为0
                    "     AVG(COALESCE(tx.numOutputs, 0)) AS avgOutputs, " +
                    "     AVG(duration.between(datetime(tx.time), datetime($currentTime)).seconds) AS avgTimeSpan " +
                    "RETURN addr, avgAmount, stdAmount, avgFee, avgInputs, avgOutputs, avgTimeSpan";

            Map<String, Object> params = new HashMap<>();
            params.put("addresses", new ArrayList<>(addresses));
            params.put("startTime", startTime.toString());
            params.put("endTime", endTime.toString());
            params.put("currentTime", currentTime.toString());

            Iterable<Map<String, Object>> result = session.query(cypher, params);
            Map<String, NeighborStats> statsMap = new HashMap<>();
            for (Map<String, Object> row : result) {
                String addr = (String) row.get("addr");
                NeighborStats stats = new NeighborStats();
                stats.setAvgAmount(safeGetDouble(row.get("avgAmount")));
                stats.setStdAmount(safeGetDouble(row.get("stdAmount")));
                stats.setAvgFee(safeGetDouble(row.get("avgFee")));
                stats.setAvgInputs(safeGetDouble(row.get("avgInputs")));
                stats.setAvgOutputs(safeGetDouble(row.get("avgOutputs")));
                stats.setAvgTimeSpan(safeGetDouble(row.get("avgTimeSpan")));
                statsMap.put(addr, stats);
            }
            return statsMap;
        } finally {
            closeSession(session);
        }
    }

    private Double safeGetDouble(Object obj) {
        if (obj == null) return 0.0;
        if (obj instanceof Number) return ((Number) obj).doubleValue();
        try {
            return Double.parseDouble(obj.toString());
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }

    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findBTCAddressesWithinNHops(String address, Integer maxHops, Integer limit) {
        try {
            if (address == null || address.isEmpty()) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            int effectiveMaxHops = getConfiguredMaxHops(maxHops);
            // 优先使用用户指定的 limit，否则使用动态限制
            int effectiveLimit = (limit != null && limit > 0) 
                ? getConfiguredNHopsLimit(limit) 
                : getDynamicNHopsLimit(maxHops);

            Session session = getSession();
            try {
                String incomeQuery = "MATCH path = (income:Address)-[:SPENT]->(tx:Transaction)-[:OUTPUT]->(start:Address {address: $address}) " +
                        "WHERE start <> income " +
                        "WITH path, nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | CASE WHEN 'Address' IN labels(n) THEN {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BTC'), first_seen: coalesce(n.first_seen, ''), last_seen: coalesce(n.last_seen, ''), type: 'address'} ELSE {txHash: n.txHash, blockHeight: n.blockHeight, time: coalesce(n.time, ''), type: 'transaction'} END] AS nodeData, " +
                        "     [r IN relList | {amount: coalesce(toFloat(r.amount), 0.0), index: coalesce(r.index, 0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData, 'income' as direction LIMIT " + effectiveLimit;

                String outcomeQuery = "MATCH path = (start:Address {address: $address})-[:SPENT]->(tx:Transaction)-[:OUTPUT]->(outcome:Address) " +
                        "WHERE start <> outcome " +
                        "WITH path, nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | CASE WHEN 'Address' IN labels(n) THEN {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BTC'), first_seen: coalesce(n.first_seen, ''), last_seen: coalesce(n.last_seen, ''), type: 'address'} ELSE {txHash: n.txHash, blockHeight: n.blockHeight, time: coalesce(n.time, ''), type: 'transaction'} END] AS nodeData, " +
                        "     [r IN relList | {amount: coalesce(toFloat(r.amount), 0.0), index: coalesce(r.index, 0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData, 'outcome' as direction LIMIT " + effectiveLimit;

                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                Iterable<Map<String, Object>> incomeQueryResult;
                try {
                    incomeQueryResult = session.query(incomeQuery, params);
                } catch (Exception e) {
                    log.error("执行BTC收入侧N跳查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                Iterable<Map<String, Object>> outcomeQueryResult;
                try {
                    outcomeQueryResult = session.query(outcomeQuery, params);
                } catch (Exception e) {
                    log.error("执行BTC支出侧N跳查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                Map<String, Object> graphDic = new HashMap<>();
                List<Map<String, Object>> allNodeList = new ArrayList<>();
                List<Map<String, Object>> allEdgeList = new ArrayList<>();
                int totalTxCount = 0;
                String firstTime = null;
                String latestTime = null;
                String addressFirstTime = null;
                String addressLatestTime = null;

                List<List<Map<String, Object>>> allPaths = new ArrayList<>();
                List<List<Map<String, Object>>> allPathRels = new ArrayList<>();
                List<List<String>> allPathTimes = new ArrayList<>();
                List<String> allPathDirections = new ArrayList<>();

                for (Map<String, Object> result : incomeQueryResult) {
                    List<Map<String, Object>> nodeData = extractNodeData(result.get("nodeData"));
                    List<Map<String, Object>> relData = extractRelData(result.get("relData"));
                    String direction = (String) result.get("direction");

                    List<String> times = new ArrayList<>();
                    for (Map<String, Object> rel : relData) {
                        Object timeObj = rel.get("time");
                        String time = timeObj != null ? timeObj.toString() : "";
                        times.add(GraphFormatUtils.parseAndFormatTimestamp(time));
                    }

                    allPaths.add(nodeData);
                    allPathRels.add(relData);
                    allPathTimes.add(times);
                    allPathDirections.add(direction);

                    totalTxCount += relData.size();
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

                for (Map<String, Object> result : outcomeQueryResult) {
                    List<Map<String, Object>> nodeData = extractNodeData(result.get("nodeData"));
                    List<Map<String, Object>> relData = extractRelData(result.get("relData"));
                    String direction = (String) result.get("direction");

                    List<String> times = new ArrayList<>();
                    for (Map<String, Object> rel : relData) {
                        Object timeObj = rel.get("time");
                        String time = timeObj != null ? timeObj.toString() : "";
                        times.add(GraphFormatUtils.parseAndFormatTimestamp(time));
                    }

                    allPaths.add(nodeData);
                    allPathRels.add(relData);
                    allPathTimes.add(times);
                    allPathDirections.add(direction);

                    totalTxCount += relData.size();
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

                Map<String, Integer> nodeLayers = calculateBTCNodeLayers(allPaths, allPathDirections, address);

                Map<String, String> globalAddressToId = new HashMap<>();
                Map<String, String> globalTxHashToId = new HashMap<>();

                String startNodeId = address;
                Map<String, Object> startNodeItem = new HashMap<>();
                startNodeItem.put("id", startNodeId);
                startNodeItem.put("label", GraphFormatUtils.shortenAddress(address));
                startNodeItem.put("title", address);
                startNodeItem.put("addr", address);
                startNodeItem.put("layer", 0);
                globalAddressToId.put(address, startNodeId);
                allNodeList.add(startNodeItem);

                for (List<Map<String, Object>> nodeData : allPaths) {
                    for (int i = 0; i < nodeData.size(); i++) {
                        Map<String, Object> nodeMap = nodeData.get(i);
                        String nodeType = nodeMap.get("type") != null ? nodeMap.get("type").toString() : "address";

                        if ("transaction".equals(nodeType)) {
                            String txHash = nodeMap.get("txHash") != null ? nodeMap.get("txHash").toString() : "";
                            if (txHash.isEmpty()) continue;

                            if (!globalTxHashToId.containsKey(txHash)) {
                                globalTxHashToId.put(txHash, txHash);

                                Map<String, Object> txNodeItem = new HashMap<>();
                                txNodeItem.put("id", txHash);
                                txNodeItem.put("label", GraphFormatUtils.shortenAddress(txHash));
                                txNodeItem.put("title", txHash);
                                txNodeItem.put("type", "transaction");
                                txNodeItem.put("txHash", txHash);
                                txNodeItem.put("layer", nodeLayers.getOrDefault(txHash, 0));

                                Object blockHeightObj = nodeMap.get("blockHeight");
                                if (blockHeightObj != null) {
                                    txNodeItem.put("blockHeight", blockHeightObj);
                                }

                                Object timeObj = nodeMap.get("time");
                                if (timeObj != null && !timeObj.toString().isEmpty()) {
                                    String txTime = GraphFormatUtils.parseAndFormatTimestamp(timeObj.toString());
                                    txNodeItem.put("time", txTime);
                                }

                                boolean exists = false;
                                for (Map<String, Object> existingNode : allNodeList) {
                                    if ("transaction".equals(existingNode.get("type")) && txHash.equals(existingNode.get("txHash"))) {
                                        exists = true;
                                        break;
                                    }
                                }
                                if (!exists) {
                                    allNodeList.add(txNodeItem);
                                }
                            }
                        } else {
                            String nodeAddress = nodeMap.get("address") != null ? nodeMap.get("address").toString() : null;
                            if (nodeAddress == null || nodeAddress.equals(address)) {
                                continue;
                            }

                            Object riskLevelObj = nodeMap.get("risk_level");
                            Integer riskLevel = 0;
                            if (riskLevelObj != null) {
                                if (riskLevelObj instanceof Integer) {
                                    riskLevel = (Integer) riskLevelObj;
                                } else if (riskLevelObj instanceof Number) {
                                    riskLevel = ((Number) riskLevelObj).intValue();
                                }
                            }

                            Object firstSeenObj = nodeMap.get("first_seen");
                            String firstSeen = firstSeenObj != null ? firstSeenObj.toString() : "";
                            if (firstSeen != null && !firstSeen.isEmpty() && (addressFirstTime == null || firstSeen.compareTo(addressFirstTime) < 0)) {
                                addressFirstTime = firstSeen;
                            }

                            Object lastSeenObj = nodeMap.get("last_seen");
                            String lastSeen = lastSeenObj != null ? lastSeenObj.toString() : "";
                            if (lastSeen != null && !lastSeen.isEmpty() && (addressLatestTime == null || lastSeen.compareTo(addressLatestTime) > 0)) {
                                addressLatestTime = lastSeen;
                            }

                            String nodeId = globalAddressToId.get(nodeAddress);
                            if (nodeId == null) {
                                nodeId = nodeAddress;
                                globalAddressToId.put(nodeAddress, nodeId);

                                Map<String, Object> nodeItem = new HashMap<>();
                                nodeItem.put("id", nodeId);
                                nodeItem.put("label", GraphFormatUtils.shortenAddress(nodeAddress));
                                nodeItem.put("title", nodeAddress);
                                nodeItem.put("addr", nodeAddress);
                                nodeItem.put("layer", nodeLayers.getOrDefault(nodeAddress, 0));

                                if (riskLevel > 0) {
                                    nodeItem.put("malicious", riskLevel);
                                }

                                boolean exists = false;
                                for (Map<String, Object> existingNode : allNodeList) {
                                    if (nodeAddress.equals(existingNode.get("addr"))) {
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
                }

                for (int pathIdx = 0; pathIdx < allPaths.size(); pathIdx++) {
                    List<Map<String, Object>> nodeData = allPaths.get(pathIdx);
                    List<Map<String, Object>> relData = allPathRels.get(pathIdx);
                    List<String> times = allPathTimes.get(pathIdx);
                    String direction = allPathDirections.get(pathIdx);

                    for (int i = 0; i < nodeData.size() - 1 && i < relData.size(); i++) {
                        Map<String, Object> fromNodeData = nodeData.get(i);
                        Map<String, Object> toNodeData = nodeData.get(i + 1);
                        Map<String, Object> relItem = relData.get(i);

                        String fromId;
                        String toId;
                        String fromType = fromNodeData.get("type") != null ? fromNodeData.get("type").toString() : "address";
                        String toType = toNodeData.get("type") != null ? toNodeData.get("type").toString() : "address";

                        if ("transaction".equals(fromType)) {
                            fromId = fromNodeData.get("txHash") != null ? fromNodeData.get("txHash").toString() : "";
                        } else {
                            fromId = fromNodeData.get("address") != null ? fromNodeData.get("address").toString() : "";
                        }

                        if ("transaction".equals(toType)) {
                            toId = toNodeData.get("txHash") != null ? toNodeData.get("txHash").toString() : "";
                        } else {
                            toId = toNodeData.get("address") != null ? toNodeData.get("address").toString() : "";
                        }

                        if (fromId.isEmpty() || toId.isEmpty()) continue;

                        boolean edgeExists = allEdgeList.stream()
                            .anyMatch(edge -> fromId.equals(edge.get("from")) && toId.equals(edge.get("to")));

                        if (!edgeExists) {
                            Map<String, Object> linkItem = new HashMap<>();
                            linkItem.put("from", fromId);
                            linkItem.put("to", toId);

                            Double amount = safeGetDouble(relItem.get("amount"));
                            linkItem.put("label", GraphFormatUtils.formatAmountLabel(amount, "BTC"));
                            linkItem.put("val", amount);

                            String txTime = "";
                            List<String> txHashList = new ArrayList<>();

                            if ("transaction".equals(fromType)) {
                                Object txTimeObj = fromNodeData.get("time");
                                if (txTimeObj != null && !txTimeObj.toString().isEmpty()) {
                                    txTime = GraphFormatUtils.parseAndFormatTimestamp(txTimeObj.toString());
                                }
                                String txHash = fromNodeData.get("txHash") != null ? fromNodeData.get("txHash").toString() : "";
                                if (!txHash.isEmpty()) {
                                    txHashList.add(txHash);
                                }
                            } else if ("transaction".equals(toType)) {
                                Object txTimeObj = toNodeData.get("time");
                                if (txTimeObj != null && !txTimeObj.toString().isEmpty()) {
                                    txTime = GraphFormatUtils.parseAndFormatTimestamp(txTimeObj.toString());
                                }
                                String txHash = toNodeData.get("txHash") != null ? toNodeData.get("txHash").toString() : "";
                                if (!txHash.isEmpty()) {
                                    txHashList.add(txHash);
                                }
                            }

                            linkItem.put("tx_time", txTime);
                            linkItem.put("tx_hash_list", txHashList);

                            allEdgeList.add(linkItem);
                        }
                    }
                }

                graphDic.put("node_list", allNodeList);
                graphDic.put("edge_list", allEdgeList);
                graphDic.put("tx_count", totalTxCount);
                graphDic.put("first_tx_time", firstTime != null ? firstTime : "");
                graphDic.put("latest_tx_time", latestTime != null ? latestTime : "");
                graphDic.put("address_first_tx_time", addressFirstTime != null ? GraphFormatUtils.parseAndFormatTimestamp(addressFirstTime) : "");
                graphDic.put("address_latest_tx_time", addressLatestTime != null ? GraphFormatUtils.parseAndFormatTimestamp(addressLatestTime) : "");

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
            log.error("查找BTC N跳内地址失败: {}", address, e);
            return ApiResponse.error(500, "查找地址失败: " + e.getMessage());
        }
    }

    private List<Map<String, Object>> extractNodeData(Object rawNodeDataObj) {
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
            }
        }
        return nodeData;
    }

    private List<Map<String, Object>> extractRelData(Object rawRelDataObj) {
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
            }
        }
        return relData;
    }

    private Map<String, Integer> calculateBTCNodeLayers(List<List<Map<String, Object>>> allPaths, List<String> allPathDirections, String centerAddress) {
        Map<String, Integer> nodeLayers = new HashMap<>();
        nodeLayers.put(centerAddress, 0);

        for (int pathIdx = 0; pathIdx < allPaths.size(); pathIdx++) {
            List<Map<String, Object>> nodeData = allPaths.get(pathIdx);
            String direction = allPathDirections.get(pathIdx);

            if ("income".equals(direction)) {
                // 收入侧路径：[上游支出方, Transaction, 中心地址]
                // layer分布：上游支出方=-2, Transaction=-1, 中心地址=0
                for (int i = 0; i < nodeData.size(); i++) {
                    Map<String, Object> nodeMap = nodeData.get(i);
                    String nodeType = nodeMap.get("type") != null ? nodeMap.get("type").toString() : "address";

                    int layer = -2 + i;

                    if ("transaction".equals(nodeType)) {
                        String txHash = nodeMap.get("txHash") != null ? nodeMap.get("txHash").toString() : "";
                        if (!txHash.isEmpty()) {
                            nodeLayers.put(txHash, layer);
                        }
                    } else {
                        String nodeAddress = nodeMap.get("address") != null ? nodeMap.get("address").toString() : "";
                        if (!nodeAddress.isEmpty()) {
                            if (nodeAddress.equals(centerAddress)) {
                                nodeLayers.put(nodeAddress, 0);
                            } else {
                                nodeLayers.put(nodeAddress, layer);
                            }
                        }
                    }
                }
            } else {
                // 支出侧路径：[中心地址, Transaction, 下游收入方]
                // layer分布：中心地址=0, Transaction=1, 下游收入方=2
                for (int i = 0; i < nodeData.size(); i++) {
                    Map<String, Object> nodeMap = nodeData.get(i);
                    String nodeType = nodeMap.get("type") != null ? nodeMap.get("type").toString() : "address";

                    int layer = i;

                    if ("transaction".equals(nodeType)) {
                        String txHash = nodeMap.get("txHash") != null ? nodeMap.get("txHash").toString() : "";
                        if (!txHash.isEmpty()) {
                            nodeLayers.put(txHash, layer);
                        }
                    } else {
                        String nodeAddress = nodeMap.get("address") != null ? nodeMap.get("address").toString() : "";
                        if (!nodeAddress.isEmpty()) {
                            if (nodeAddress.equals(centerAddress)) {
                                nodeLayers.put(nodeAddress, 0);
                            } else {
                                nodeLayers.put(nodeAddress, layer);
                            }
                        }
                    }
                }
            }
        }

        return nodeLayers;
    }

    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findBTCNhopTransactionPath(String fromAddress, String toAddress, Integer maxHops, Integer limit) {
        try {
            if (fromAddress == null || fromAddress.isEmpty() || toAddress == null || toAddress.isEmpty()) {
                return ApiResponse.error(400, "起始地址和目标地址不能为空");
            }

            int effectiveMaxHops = getConfiguredMaxHops(maxHops);
            // 优先使用用户指定的 limit，否则使用动态限制
            int effectiveLimit = (limit != null && limit > 0) 
                ? getConfiguredPathLimit(limit) 
                : getDynamicPathLimit(maxHops);

            Session session = getSession();
            try {
                String pathQuery = "MATCH path = (from:Address {address: $fromAddress})" +
                        "-[:SPENT|OUTPUT*1.." + effectiveMaxHops + "]->(to:Address {address: $toAddress}) " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {" +
                        "   address: coalesce(n.address, ''), " +
                        "   txHash: coalesce(n.txHash, ''), " +
                        "   type: CASE WHEN 'Transaction' IN labels(n) THEN 'transaction' ELSE 'address' END, " +
                        "   risk_level: coalesce(n.risk_level, 0), " +
                        "   blockHeight: coalesce(n.blockHeight, 0), " +
                        "   time: coalesce(n.time, '') " +
                        "}] AS nodeData, " +
                        "     [r IN relList | {" +
                        "   from: coalesce(startNode(r).address, startNode(r).txHash, ''), " +
                        "   to: coalesce(endNode(r).address, endNode(r).txHash, ''), " +
                        "   type: type(r), " +
                        "   amount: coalesce(toFloat(r.amount), 0.0), " +
                        "   time: coalesce(r.time, '')" +
                        "}] AS relData " +
                        "RETURN nodeData, relData LIMIT " + effectiveLimit;

                Map<String, Object> params = new HashMap<>();
                params.put("fromAddress", fromAddress);
                params.put("toAddress", toAddress);

                Iterable<Map<String, Object>> queryResult;
                try {
                    queryResult = session.query(pathQuery, params);
                } catch (Exception e) {
                    log.error("执行BTC交易路径查询时发生错误: {}", e.getMessage(), e);
                    return ApiResponse.error(500, "查询执行失败: " + e.getMessage());
                }

                Map<String, Object> graphDic = new HashMap<>();
                List<Map<String, Object>> allNodeList = new ArrayList<>();
                List<Map<String, Object>> allEdgeList = new ArrayList<>();
                int totalTxCount = 0;
                String firstTime = null;
                String latestTime = null;

                List<List<Map<String, Object>>> allPaths = new ArrayList<>();
                List<List<Map<String, Object>>> allPathRels = new ArrayList<>();
                List<List<String>> allPathTimes = new ArrayList<>();

                for (Map<String, Object> result : queryResult) {
                    List<Map<String, Object>> nodeData = extractNodeData(result.get("nodeData"));
                    List<Map<String, Object>> relData = extractRelData(result.get("relData"));

                    List<String> times = new ArrayList<>();
                    for (Map<String, Object> nodeMap : nodeData) {
                        String nodeType = nodeMap.get("type") != null ? nodeMap.get("type").toString() : "address";
                        if ("transaction".equals(nodeType)) {
                            Object timeObj = nodeMap.get("time");
                            String time = timeObj != null ? timeObj.toString() : "";
                            times.add(GraphFormatUtils.parseAndFormatTimestamp(time));
                        }
                    }

                    allPaths.add(nodeData);
                    allPathRels.add(relData);
                    allPathTimes.add(times);

                    totalTxCount += relData.size();
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

                if (allPaths.isEmpty()) {
                    graphDic.put("node_list", new ArrayList<>());
                    graphDic.put("edge_list", new ArrayList<>());
                    graphDic.put("tx_count", 0);
                    graphDic.put("first_tx_time", "");
                    graphDic.put("latest_tx_time", "");
                    graphDic.put("from_address", fromAddress);
                    graphDic.put("to_address", toAddress);
                    return ApiResponse.success(graphDic, 0L);
                }

                Map<String, Integer> nodeLayers = calculateBTCPathLayers(allPaths, fromAddress, toAddress);

                for (List<Map<String, Object>> nodeData : allPaths) {
                    for (int i = 0; i < nodeData.size(); i++) {
                        Map<String, Object> nodeMap = nodeData.get(i);
                        String nodeType = nodeMap.get("type") != null ? nodeMap.get("type").toString() : "address";

                        if ("transaction".equals(nodeType)) {
                            String txHash = nodeMap.get("txHash") != null ? nodeMap.get("txHash").toString() : "";
                            if (txHash.isEmpty()) continue;

                            boolean exists = false;
                            for (Map<String, Object> existingNode : allNodeList) {
                                if ("transaction".equals(existingNode.get("type")) && txHash.equals(existingNode.get("txHash"))) {
                                    exists = true;
                                    break;
                                }
                            }

                            if (!exists) {
                                Map<String, Object> txNodeItem = new HashMap<>();
                                txNodeItem.put("id", txHash);
                                txNodeItem.put("label", GraphFormatUtils.shortenAddress(txHash));
                                txNodeItem.put("title", txHash);
                                txNodeItem.put("type", "transaction");
                                txNodeItem.put("txHash", txHash);
                                txNodeItem.put("layer", nodeLayers.getOrDefault(txHash, 0));

                                Object blockHeightObj = nodeMap.get("blockHeight");
                                if (blockHeightObj != null) {
                                    txNodeItem.put("blockHeight", blockHeightObj);
                                }

                                Object timeObj = nodeMap.get("time");
                                if (timeObj != null && !timeObj.toString().isEmpty()) {
                                    String txTime = GraphFormatUtils.parseAndFormatTimestamp(timeObj.toString());
                                    txNodeItem.put("time", txTime);
                                }

                                allNodeList.add(txNodeItem);
                            }
                        } else {
                            String nodeAddress = nodeMap.get("address") != null ? nodeMap.get("address").toString() : null;
                            if (nodeAddress == null) {
                                continue;
                            }

                            boolean exists = false;
                            for (Map<String, Object> existingNode : allNodeList) {
                                if (nodeAddress.equals(existingNode.get("addr"))) {
                                    exists = true;
                                    break;
                                }
                            }

                            if (!exists) {
                                Object riskLevelObj = nodeMap.get("risk_level");
                                Integer riskLevel = 0;
                                if (riskLevelObj != null) {
                                    if (riskLevelObj instanceof Integer) {
                                        riskLevel = (Integer) riskLevelObj;
                                    } else if (riskLevelObj instanceof Number) {
                                        riskLevel = ((Number) riskLevelObj).intValue();
                                    }
                                }

                                Map<String, Object> nodeItem = new HashMap<>();
                                nodeItem.put("id", nodeAddress);
                                nodeItem.put("label", GraphFormatUtils.shortenAddress(nodeAddress));
                                nodeItem.put("title", nodeAddress);
                                nodeItem.put("addr", nodeAddress);
                                nodeItem.put("type", "address");
                                nodeItem.put("layer", nodeLayers.getOrDefault(nodeAddress, 0));

                                if (riskLevel > 0) {
                                    nodeItem.put("malicious", riskLevel);
                                }

                                allNodeList.add(nodeItem);
                            }
                        }
                    }
                }

                for (int pathIdx = 0; pathIdx < allPaths.size(); pathIdx++) {
                    List<Map<String, Object>> nodeData = allPaths.get(pathIdx);
                    List<Map<String, Object>> relData = allPathRels.get(pathIdx);

                    for (int i = 0; i < nodeData.size() - 1 && i < relData.size(); i++) {
                        Map<String, Object> fromNodeData = nodeData.get(i);
                        Map<String, Object> toNodeData = nodeData.get(i + 1);
                        Map<String, Object> relItem = relData.get(i);

                        String fromId;
                        String toId;
                        String fromType = fromNodeData.get("type") != null ? fromNodeData.get("type").toString() : "address";
                        String toType = toNodeData.get("type") != null ? toNodeData.get("type").toString() : "address";

                        if ("transaction".equals(fromType)) {
                            fromId = fromNodeData.get("txHash") != null ? fromNodeData.get("txHash").toString() : "";
                        } else {
                            fromId = fromNodeData.get("address") != null ? fromNodeData.get("address").toString() : "";
                        }

                        if ("transaction".equals(toType)) {
                            toId = toNodeData.get("txHash") != null ? toNodeData.get("txHash").toString() : "";
                        } else {
                            toId = toNodeData.get("address") != null ? toNodeData.get("address").toString() : "";
                        }

                        if (fromId.isEmpty() || toId.isEmpty()) continue;

                        boolean edgeExists = allEdgeList.stream()
                            .anyMatch(edge -> fromId.equals(edge.get("from")) && toId.equals(edge.get("to")));

                        if (!edgeExists) {
                            Map<String, Object> linkItem = new HashMap<>();
                            linkItem.put("from", fromId);
                            linkItem.put("to", toId);

                            Double amount = safeGetDouble(relItem.get("amount"));
                            linkItem.put("label", GraphFormatUtils.formatAmountLabel(amount, "BTC"));
                            linkItem.put("val", amount);

                            String txTime = "";
                            List<String> txHashList = new ArrayList<>();
                            if ("transaction".equals(fromType)) {
                                Object txTimeObj = fromNodeData.get("time");
                                if (txTimeObj != null && !txTimeObj.toString().isEmpty()) {
                                    txTime = GraphFormatUtils.parseAndFormatTimestamp(txTimeObj.toString());
                                }
                                String txHash = fromNodeData.get("txHash") != null ? fromNodeData.get("txHash").toString() : "";
                                if (!txHash.isEmpty()) {
                                    txHashList.add(txHash);
                                }
                            } else if ("transaction".equals(toType)) {
                                Object txTimeObj = toNodeData.get("time");
                                if (txTimeObj != null && !txTimeObj.toString().isEmpty()) {
                                    txTime = GraphFormatUtils.parseAndFormatTimestamp(txTimeObj.toString());
                                }
                                String txHash = toNodeData.get("txHash") != null ? toNodeData.get("txHash").toString() : "";
                                if (!txHash.isEmpty()) {
                                    txHashList.add(txHash);
                                }
                            }
                            linkItem.put("tx_time", txTime);
                            linkItem.put("tx_hash_list", txHashList);

                            allEdgeList.add(linkItem);
                        }
                    }
                }

                graphDic.put("node_list", allNodeList);
                graphDic.put("edge_list", allEdgeList);
                graphDic.put("tx_count", totalTxCount);
                graphDic.put("first_tx_time", firstTime != null ? firstTime : "");
                graphDic.put("latest_tx_time", latestTime != null ? latestTime : "");
                graphDic.put("from_address", fromAddress);
                graphDic.put("to_address", toAddress);

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
            log.error("查找BTC交易路径失败: {} -> {}", fromAddress, toAddress, e);
            return ApiResponse.error(500, "查找交易路径失败: " + e.getMessage());
        }
    }

    private Map<String, Integer> calculateBTCPathLayers(List<List<Map<String, Object>>> allPaths, String fromAddress, String toAddress) {
        Map<String, Integer> nodeLayers = new HashMap<>();
        nodeLayers.put(fromAddress, 0);

        if (allPaths == null || allPaths.isEmpty()) {
            nodeLayers.put(toAddress, 1);
            return nodeLayers;
        }

        int maxLayer = 0;
        for (List<Map<String, Object>> path : allPaths) {
            int fromIndex = -1;
            for (int i = 0; i < path.size(); i++) {
                Map<String, Object> nodeMap = path.get(i);
                String nodeType = nodeMap.get("type") != null ? nodeMap.get("type").toString() : "address";
                String nodeId;

                if ("transaction".equals(nodeType)) {
                    nodeId = nodeMap.get("txHash") != null ? nodeMap.get("txHash").toString() : "";
                } else {
                    nodeId = nodeMap.get("address") != null ? nodeMap.get("address").toString() : "";
                }

                if (nodeId.equals(fromAddress)) {
                    fromIndex = i;
                    break;
                }
            }

            if (fromIndex != -1) {
                for (int i = 0; i < path.size(); i++) {
                    Map<String, Object> nodeMap = path.get(i);
                    String nodeType = nodeMap.get("type") != null ? nodeMap.get("type").toString() : "address";
                    String nodeId;

                    if ("transaction".equals(nodeType)) {
                        nodeId = nodeMap.get("txHash") != null ? nodeMap.get("txHash").toString() : "";
                    } else {
                        nodeId = nodeMap.get("address") != null ? nodeMap.get("address").toString() : "";
                    }

                    if (nodeId.isEmpty()) continue;

                    int layer = i - fromIndex;
                    if (Math.abs(layer) > maxLayer) {
                        maxLayer = Math.abs(layer);
                    }

                    if (!nodeLayers.containsKey(nodeId) || Math.abs(layer) < Math.abs(nodeLayers.get(nodeId))) {
                        nodeLayers.put(nodeId, layer);
                    }
                }
            }
        }

        if (!nodeLayers.containsKey(toAddress)) {
            nodeLayers.put(toAddress, maxLayer);
        }

        Set<Integer> uniqueLayers = new HashSet<>(nodeLayers.values());
        List<Integer> sortedLayers = new ArrayList<>(uniqueLayers);
        Collections.sort(sortedLayers);
        Map<Integer, Integer> layerMap = new HashMap<>();
        for (int i = 0; i < sortedLayers.size(); i++) {
            layerMap.put(sortedLayers.get(i), i);
        }

        Map<String, Integer> remappedLayers = new HashMap<>();
        for (Map.Entry<String, Integer> entry : nodeLayers.entrySet()) {
            remappedLayers.put(entry.getKey(), layerMap.get(entry.getValue()));
        }

        return remappedLayers;
    }


}