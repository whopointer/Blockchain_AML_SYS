package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.util.GraphFormatUtils;
import com.seecoder.DataProcessing.util.GraphLayerCalculator;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.session.Session;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class GraphAddressService extends AbstractGraphService {

    /**
     * 查找两个地址之间的交易路径
     * @param fromAddress 起始地址
     * @param toAddress 目标地址
     * @return 包含路径节点、边、交易信息的响应
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findNhopTransactionPath(String fromAddress, String toAddress) {
        try {
            if (fromAddress == null || toAddress == null) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            Session session = getSession();
            try {
                // 去除maxHops限制，使用默认的较小跳数限制避免查询时间过长
                String pathQuery = "MATCH path = (a:Address {address: $fromAddress})" +
                        "-[:TRANSFER*1..5]->(b:Address {address: $toAddress}) " +
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
     * 查找指定地址N跳内的所有关联地址
     * @param address 中心地址
     * @param maxHops 最大跳数，默认1，范围1-6
     * @return 包含节点列表、边列表、交易计数等信息的响应
     */
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findAddressesWithinNHops(String address, Integer maxHops) {
        try {
            if (address == null || address.isEmpty()) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            if (maxHops == null || maxHops <= 0 || maxHops > 6) {
                maxHops = 1; // 默认为1
            }

            Session session = getSession();
            try {
                // 分别查询收入侧（入度）和支出侧（出度）的路径
                String incomeQuery = "MATCH path = (start:Address {address: $address})<-[:TRANSFER*1.." + maxHops + "]-(income:Address) " +
                        "WHERE start <> income " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BNB'), first_seen: coalesce(n.first_seen, ''), last_seen: coalesce(n.last_seen, '')} ] AS nodeData, " +
                        "     [r IN relList | {tx_hash: coalesce(r.tx_hash, ''), amount: coalesce(toFloat(r.amount), 0.0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData, 'income' as direction LIMIT 50";

                String outcomeQuery = "MATCH path = (start:Address {address: $address})-[:TRANSFER*1.." + maxHops + "]->(outcome:Address) " +
                        "WHERE start <> outcome " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BNB'), first_seen: coalesce(n.first_seen, ''), last_seen: coalesce(n.last_seen, '')} ] AS nodeData, " +
                        "     [r IN relList | {tx_hash: coalesce(r.tx_hash, ''), amount: coalesce(toFloat(r.amount), 0.0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData, 'outcome' as direction LIMIT 50";

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
                
                // 计算节点层级，起始节点为0，收入侧为负值，支出侧为正值
                Map<String, Integer> nodeLayers = GraphLayerCalculator.calculateNodeLayersWithDirection(allPaths, allPathDirections, address);
                
                Map<String, String> globalAddressToId = new HashMap<>();
                
                // 添加起始节点
                String startNodeId = java.util.UUID.randomUUID().toString();
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
                            // 生成新的ID并记录到全局映射
                            nodeId = java.util.UUID.randomUUID().toString();
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
                                Map<String, Object> linkItem = new HashMap<>();
                                linkItem.put("from", fromAddr);
                                linkItem.put("to", toAddr);
                                
                                String chain = fromNodeData != null ? fromNodeData.get("chain").toString() : "BNB";
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
                graphDic.put("first_tx_time", firstTime != null ? firstTime : "");
                graphDic.put("latest_tx_time", latestTime != null ? latestTime : "");
                graphDic.put("address_first_tx_time", addressFirstTime != null ? addressFirstTime : "");
                graphDic.put("address_latest_tx_time", addressLatestTime != null ? addressLatestTime : "");

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
    



}