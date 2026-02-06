package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.po.graph.AddressNode;
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

    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> findNhopTransactionPath(String fromAddress, String toAddress, Integer maxHops) {
        try {
            if (fromAddress == null || toAddress == null) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            if (maxHops == null || maxHops <= 0 || maxHops > 10) {
                maxHops = 5;
            }

            Session session = getSession();
            try {
                // 修改：使用更简单的查询方式来避免OGM映射问题
                // 首先查找路径
                String pathQuery = "MATCH path = shortestPath((a:Address {address: $fromAddress})" +
                        "-[:TRANSFER*1.." + maxHops + "]->(b:Address {address: $toAddress})) " +
                        "WITH nodes(path) AS nodeList, relationships(path) AS relList " +
                        "WITH [n IN nodeList | {address: n.address, risk_level: coalesce(n.risk_level, 0), chain: coalesce(n.chain, 'BNB')} ] AS nodeData, " +
                        "     [r IN relList | {tx_hash: coalesce(r.tx_hash, ''), amount: coalesce(toFloat(r.amount), 0.0), time: coalesce(r.time, '')}] AS relData " +
                        "RETURN nodeData, relData LIMIT 1";

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

                // 初始化默认返回值
                Map<String, Object> graphDic = new java.util.HashMap<>();
                graphDic.put("node_list", new java.util.ArrayList<>());
                graphDic.put("edge_list", new java.util.ArrayList<>());
                graphDic.put("tx_count", 0);
                graphDic.put("first_tx_datetime", "");
                graphDic.put("latest_tx_datetime", "");
                graphDic.put("address_first_tx_datetime", "");
                graphDic.put("address_latest_tx_datetime", "");

                for (Map<String, Object> result : queryResult) {
                    // 添加调试日志来查看原始查询结果
                    log.debug("Raw query result: {}", result);
                    
                    // 节点列表
                    List<Map<String, Object>> nodeList = new java.util.ArrayList<>();
                    
                    // 安全转换节点列表 - 处理Neo4j OGM映射问题
                    Object rawNodeDataObj = result.get("nodeData");
                    List<Map<String, Object>> nodeData = new ArrayList<>();
                    if (rawNodeDataObj instanceof List) {
                        nodeData = (List<Map<String, Object>>) rawNodeDataObj;
                    } else if (rawNodeDataObj != null) {
                        // 如果是数组类型（如 [Ljava.util.Collections$UnmodifiableMap;），则转换为List
                        if (rawNodeDataObj.getClass().isArray()) {
                            Object[] array = (Object[]) rawNodeDataObj;
                            for (Object obj : array) {
                                if (obj instanceof Map) {
                                    nodeData.add((Map<String, Object>) obj);
                                }
                            }
                        } else if (rawNodeDataObj instanceof Map) {
                            // 如果是单个Map，添加到列表中
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
                        // 如果是数组类型（如 [Ljava.util.Collections$UnmodifiableMap;），则转换为List
                        if (rawRelDataObj.getClass().isArray()) {
                            Object[] array = (Object[]) rawRelDataObj;
                            for (Object obj : array) {
                                if (obj instanceof Map) {
                                    relData.add((Map<String, Object>) obj);
                                }
                            }
                        } else if (rawRelDataObj instanceof Map) {
                            // 如果是单个Map，添加到列表中
                            relData.add((Map<String, Object>) rawRelDataObj);
                        } else {
                            log.warn("Unexpected data type for relData: {}", rawRelDataObj.getClass().getName());
                        }
                    }
                    
                    // 提取单独的列表
                    List<String> txHashes = new ArrayList<>();
                    List<Double> amounts = new ArrayList<>();
                    List<String> times = new ArrayList<>();
                    
                    for (Map<String, Object> rel : relData) {
                        // 安全地转换 tx_hash 为字符串
                        Object txHashObj = rel.get("tx_hash");
                        String txHash = txHashObj != null ? txHashObj.toString() : "";
                        txHashes.add(txHash);
                        
                        // 安全地转换 amount 为双精度浮点数
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
                        
                        // 安全地转换 time 为字符串
                        Object timeObj = rel.get("time");
                        String time = timeObj != null ? timeObj.toString() : "";
                        times.add(time);
                    }
                    
                    // 创建地址到ID的映射
                    Map<String, String> addressToId = new java.util.HashMap<>();
                    
                    for (int i = 0; i < nodeData.size(); i++) {
                        // 从nodeData中提取数据
                        Map<String, Object> nodeMap = nodeData.get(i);
                        // 安全地转换 address 为字符串
                        Object addressObj = nodeMap.get("address");
                        String address = addressObj != null ? addressObj.toString() : null;
                        
                        Object riskLevelObj = nodeMap.get("risk_level");
                        Integer riskLevel = 0;
                        
                        if (riskLevelObj != null) {
                            if (riskLevelObj instanceof Integer) {
                                riskLevel = (Integer) riskLevelObj;
                            } else if (riskLevelObj instanceof Number) {
                                riskLevel = ((Number) riskLevelObj).intValue();
                            } else {
                                // 如果风险等级不是数字类型，尝试转换
                                try {
                                    riskLevel = Integer.parseInt(riskLevelObj.toString());
                                } catch (NumberFormatException e) {
                                    log.warn("Cannot parse risk level: {}", riskLevelObj);
                                }
                            }
                        }
                        
                        if (address == null) {
                            continue; // 跳过没有地址的节点
                        }
                        
                        // 生成唯一ID
                        String nodeId = java.util.UUID.randomUUID().toString();
                        addressToId.put(address, nodeId);
                        
                        Map<String, Object> nodeItem = new java.util.HashMap<>();
                        nodeItem.put("id", nodeId);
                        nodeItem.put("label", shortenAddress(address)); // 缩短显示的地址
                        nodeItem.put("title", address); // 完整地址
                        nodeItem.put("addr", address);
                        
                        // 设置层级：起始地址为0层，其他节点为相对层数
                        int layer = i; // 根据位置分配层号
                        nodeItem.put("layer", layer);
                        
                        // 如果是恶意地址（风险等级 > 0），则添加恶意标记
                        if (riskLevel > 0) {
                            nodeItem.put("malicious", riskLevel);
                        }
                        
                        nodeList.add(nodeItem);
                    }
                    
                    // 边列表
                    List<Map<String, Object>> edgeList = new java.util.ArrayList<>();
                    
                    for (int i = 0; i < nodeData.size() - 1 && i < relData.size(); i++) {
                        if (i < nodeList.size() - 1) {
                            // 获取对应的节点数据以获得地址信息
                            Map<String, Object> fromNodeData = nodeData.get(i);
                            Map<String, Object> toNodeData = nodeData.get(i + 1);
                            
                            // 使用地址作为 from 和 to 的值
                            String fromAddr = fromNodeData != null ? fromNodeData.get("address").toString() : "";
                            String toAddr = toNodeData != null ? toNodeData.get("address").toString() : "";
                            
                            Map<String, Object> linkItem = new java.util.HashMap<>();
                            linkItem.put("from", fromAddr);
                            linkItem.put("to", toAddr);
                            // 获取源节点的链信息用于确定货币单位
                            String chain = fromNodeData != null ? fromNodeData.get("chain").toString() : "BNB";
                            linkItem.put("label", formatAmountLabel(amounts.get(i), chain));
                            linkItem.put("val", amounts.get(i));
                            linkItem.put("tx_time", times.get(i));
                            
                            // 将单个交易哈希包装成数组
                            List<String> txHashList = new java.util.ArrayList<>();
                            if (txHashes.get(i) != null && !txHashes.get(i).isEmpty()) {
                                txHashList.add(txHashes.get(i));
                            }
                            linkItem.put("tx_hash_list", txHashList);
                            
                            edgeList.add(linkItem);
                        }
                    }
                    
                    graphDic.put("node_list", nodeList);
                    graphDic.put("edge_list", edgeList);
                    
                    // 添加调试日志来查看处理后的结果
                    log.debug("Processed nodeData size: {}, relData size: {}", nodeData.size(), relData.size());
                    log.debug("Generated nodeList size: {}, edgeList size: {}", nodeList.size(), edgeList.size());
                    
                    // 计算一些统计数据
                    int txCount = Math.min(Math.min(txHashes.size(), amounts.size()), times.size());
                    graphDic.put("tx_count", txCount);
                    
                    // 设置时间范围
                    if (!times.isEmpty()) {
                        graphDic.put("first_tx_datetime", times.get(0));
                        graphDic.put("latest_tx_datetime", times.get(times.size() - 1));
                        graphDic.put("address_first_tx_datetime", times.get(0));
                        graphDic.put("address_latest_tx_datetime", times.get(times.size() - 1));
                    } else {
                        graphDic.put("first_tx_datetime", "");
                        graphDic.put("latest_tx_datetime", "");
                        graphDic.put("address_first_tx_datetime", "");
                        graphDic.put("address_latest_tx_datetime", "");
                    }
                    
                    break; // 只处理第一个结果
                }

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
    
    // 辅助方法：缩短地址显示
    private String shortenAddress(String address) {
        if (address == null || address.length() <= 8) {
            return address;
        }
        return address.substring(0, 5) + "..." + address.substring(address.length() - 5);
    }
    
    // 辅助方法：格式化金额标签
    private String formatAmountLabel(double amount, String chain) {
        String unit = (chain != null && chain.equalsIgnoreCase("ethereum")) ? "ETH" : "BNB";
        if (amount >= 1.0) {
            return String.format("%.3f %s", amount, unit);
        } else {
            return String.format("%.6f %s", amount, unit);
        }
    }

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

    @Transactional(transactionManager = "neo4jTransactionManager")
    public ApiResponse<Void> updateAddressRiskLevel(String address, Integer riskLevel) {
        try {
            Session session = getSession();
            try {
                // 先检查地址是否存在
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

                // 更新风险等级
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

    @Transactional(transactionManager = "neo4jTransactionManager")
    public ApiResponse<Void> tagAddress(String address, String tag) {
        try {
            Session session = getSession();
            try {
                // 先检查地址是否存在
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

                // 更新标签
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