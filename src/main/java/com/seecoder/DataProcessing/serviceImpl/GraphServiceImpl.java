// com/seecoder/DataProcessing/serviceImpl/GraphServiceImpl.java
package com.seecoder.DataProcessing.serviceImpl;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.graph.*;
import com.seecoder.DataProcessing.repository.graph.AddressNodeRepository;
import com.seecoder.DataProcessing.repository.graph.TransactionNodeRepository;
import com.seecoder.DataProcessing.repository.graph.TransferRelationRepository;
import com.seecoder.DataProcessing.service.GraphService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.model.Node;
import org.neo4j.ogm.response.model.NodeModel;
import org.neo4j.ogm.response.model.RelationshipModel;
import org.neo4j.ogm.session.Session;
import org.neo4j.ogm.session.SessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CompletableFuture;

@Slf4j
@Service
public class GraphServiceImpl implements GraphService {

    @Autowired
    private SessionFactory sessionFactory;

    @Autowired
    private AddressNodeRepository addressNodeRepository;

    @Autowired
    private TransactionNodeRepository transactionNodeRepository;

    @Autowired
    private TransferRelationRepository transferRelationRepository;

    // 获取Session
    private Session getSession() {
        return sessionFactory.openSession();
    }

    // 清理Session
    private void closeSession(Session session) {
        if (session != null) {
            session.clear();
        }
    }

    private Double convertBigDecimal(BigDecimal value) {
        return value != null ? value.doubleValue() : 0.0;
    }

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveTransactionToGraph(ChainTx chainTx) {
        Session session = null;
        try {
            log.debug("开始保存交易到图数据库: {}", chainTx.getTxHash());
            session = getSession();

            // 1. 验证必要字段
            if (chainTx.getFromAddress() == null || chainTx.getFromAddress().isEmpty()) {
                log.warn("发送地址为空，跳过保存: {}", chainTx.getTxHash());
                return;
            }

            // 2. 使用原生Cypher查询创建节点和关系
            StringBuilder cypher = new StringBuilder();
            Map<String, Object> params = new HashMap<>();

            // 基础参数
            params.put("chain", chainTx.getChain());
            params.put("txHash", chainTx.getTxHash());
            params.put("fromAddress", chainTx.getFromAddress());
            params.put("blockHeight", chainTx.getBlockHeight());
            params.put("blockTime", chainTx.getBlockTime());

            // 修改：将BigDecimal转换为Double
            Double totalInput = convertBigDecimal(chainTx.getTotalInput());
            Double totalOutput = convertBigDecimal(chainTx.getTotalOutput());
            Double fee = convertBigDecimal(chainTx.getFee());

            params.put("totalInput", totalInput);
            params.put("totalOutput", totalOutput);
            params.put("fee", fee);
            params.put("txIndex", chainTx.getTxIndex() != null ? chainTx.getTxIndex() : 0);

            // 3. 构建Cypher查询
            // 创建/更新发送地址
            cypher.append("MERGE (a1:Address {address: $fromAddress, chain: $chain}) ")
                    .append("ON CREATE SET a1.firstSeen = $blockTime, a1.lastSeen = $blockTime, ")
                    .append("a1.riskLevel = 0, a1.tag = '' ")
                    .append("ON MATCH SET a1.lastSeen = $blockTime ");

            // 创建/更新接收地址（如果存在）
            if (chainTx.getToAddress() != null && !chainTx.getToAddress().isEmpty()) {
                params.put("toAddress", chainTx.getToAddress());

                cypher.append("MERGE (a2:Address {address: $toAddress, chain: $chain}) ")
                        .append("ON CREATE SET a2.firstSeen = $blockTime, a2.lastSeen = $blockTime, ")
                        .append("a2.riskLevel = 0, a2.tag = '' ")
                        .append("ON MATCH SET a2.lastSeen = $blockTime ");
            }

            // 创建/更新交易节点
            cypher.append("MERGE (tx:Transaction {txHash: $txHash, chain: $chain}) ")
                    .append("ON CREATE SET tx.blockHeight = $blockHeight, tx.time = $blockTime, ")
                    .append("tx.totalInput = $totalInput, tx.totalOutput = $totalOutput, ")
                    .append("tx.fee = $fee ");

            // 创建关系
            if (chainTx.getToAddress() != null && !chainTx.getToAddress().isEmpty() && totalOutput > 0) {
                // TRANSFER 关系（地址->地址）
                cypher.append("WITH a1, a2, tx ")
                        .append("MERGE (a1)-[tr:TRANSFER]->(a2) ")
                        .append("ON CREATE SET tr.txHash = $txHash, tr.amount = $totalOutput, tr.time = $blockTime ")
                        .append("ON MATCH SET tr.amount = $totalOutput, tr.time = $blockTime ");

                // OUTPUT 关系（交易->地址）
                cypher.append("MERGE (tx)-[out:OUTPUT]->(a2) ")
                        .append("ON CREATE SET out.amount = $totalOutput, out.index = $txIndex ")
                        .append("ON MATCH SET out.amount = $totalOutput, out.index = $txIndex ");
            }

            // SPENT 关系（地址->交易）
            if (totalInput > 0) {
                cypher.append("WITH a1, tx ")
                        .append("MERGE (a1)-[sp:SPENT]->(tx) ")
                        .append("ON CREATE SET sp.amount = $totalInput, sp.index = $txIndex ")
                        .append("ON MATCH SET sp.amount = $totalInput, sp.index = $txIndex ");
            }

            // 4. 执行Cypher查询
            session.query(cypher.toString(), params);
            log.info("成功保存交易到图数据库: {}", chainTx.getTxHash());

        } catch (Exception e) {
            log.error("保存交易到图数据库失败: {}", chainTx.getTxHash(), e);
            // 不抛出异常，避免影响主流程
        } finally {
            if (session != null) {
                closeSession(session);
            }
        }
    }

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveTransactionsToGraph(List<ChainTx> chainTxs) {
        log.info("开始批量保存 {} 笔交易到图数据库", chainTxs.size());

        if (chainTxs == null || chainTxs.isEmpty()) {
            log.info("没有交易需要保存");
            return;
        }

        int successCount = 0;
        int failCount = 0;

        // 分批处理，每批50个（减小批次大小）
        int batchSize = 50;
        for (int i = 0; i < chainTxs.size(); i += batchSize) {
            int end = Math.min(chainTxs.size(), i + batchSize);
            List<ChainTx> batch = chainTxs.subList(i, end);

            log.info("处理批次 {}-{}", i + 1, end);

            Session session = getSession();
            try {
                for (ChainTx tx : batch) {
                    try {
                        // 使用简化版的Cypher保存
                        saveTransactionSimpleWithCypher(tx, session);
                        successCount++;

                        if (successCount % 50 == 0) {
                            log.info("已保存 {} 条记录", successCount);
                        }
                    } catch (Exception e) {
                        log.error("保存交易失败: {}", tx.getTxHash(), e);
                        failCount++;
                    }
                }
            } finally {
                closeSession(session);
            }
        }

        log.info("批量保存完成: 成功 {} 笔, 失败 {} 笔", successCount, failCount);
    }

    // 修改：简化版Cypher保存方法，避免复杂的关系映射
    private void saveTransactionSimpleWithCypher(ChainTx chainTx, Session session) {
        if (chainTx.getFromAddress() == null || chainTx.getFromAddress().isEmpty()) {
            log.warn("发送地址为空，跳过保存: {}", chainTx.getTxHash());
            return;
        }

        try {
            // 构建参数
            Map<String, Object> params = new HashMap<>();
            params.put("chain", chainTx.getChain());
            params.put("txHash", chainTx.getTxHash());
            params.put("fromAddress", chainTx.getFromAddress());
            params.put("blockHeight", chainTx.getBlockHeight());
            params.put("blockTime", chainTx.getBlockTime());

            // 转换BigDecimal为Double
            Double totalInput = convertBigDecimal(chainTx.getTotalInput());
            Double totalOutput = convertBigDecimal(chainTx.getTotalOutput());
            Double fee = convertBigDecimal(chainTx.getFee());

            params.put("totalInput", totalInput);
            params.put("totalOutput", totalOutput);
            params.put("fee", fee);
            params.put("txIndex", chainTx.getTxIndex() != null ? chainTx.getTxIndex() : 0);

            // 构建简化Cypher查询
            StringBuilder cypher = new StringBuilder();

            // 1. 创建/更新发送地址
            cypher.append("MERGE (from:Address {address: $fromAddress, chain: $chain}) ");
            cypher.append("ON CREATE SET from.firstSeen = $blockTime, from.lastSeen = $blockTime, ");
            cypher.append("from.riskLevel = 0, from.tag = '' ");
            cypher.append("ON MATCH SET from.lastSeen = $blockTime ");

            // 2. 创建交易节点
            cypher.append("MERGE (tx:Transaction {txHash: $txHash, chain: $chain}) ");
            cypher.append("ON CREATE SET tx.blockHeight = $blockHeight, tx.time = $blockTime, ");
            cypher.append("tx.totalInput = $totalInput, tx.totalOutput = $totalOutput, ");
            cypher.append("tx.fee = $fee ");

            // 3. 创建SPENT关系
            cypher.append("MERGE (from)-[:SPENT {amount: $totalInput, index: $txIndex}]->(tx) ");

            // 4. 如果有接收地址
            if (chainTx.getToAddress() != null && !chainTx.getToAddress().isEmpty()) {
                params.put("toAddress", chainTx.getToAddress());

                // 创建/更新接收地址
                cypher.append("MERGE (to:Address {address: $toAddress, chain: $chain}) ");
                cypher.append("ON CREATE SET to.firstSeen = $blockTime, to.lastSeen = $blockTime, ");
                cypher.append("to.riskLevel = 0, to.tag = '' ");
                cypher.append("ON MATCH SET to.lastSeen = $blockTime ");

                // 创建TRANSFER关系
                cypher.append("MERGE (from)-[:TRANSFER {txHash: $txHash, amount: $totalOutput, time: $blockTime}]->(to) ");

                // 创建OUTPUT关系
                cypher.append("MERGE (tx)-[:OUTPUT {amount: $totalOutput, index: $txIndex}]->(to) ");
            }

            // 执行查询
            session.query(cypher.toString(), params);

        } catch (Exception e) {
            log.error("使用Cypher保存交易失败: {}", chainTx.getTxHash(), e);
            throw e;
        }
    }

    @Override
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<List<Map<String, Object>>> findNhopTransactionPath(String fromAddress, String toAddress, Integer maxHops) {
        try {
            if (fromAddress == null || toAddress == null) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            if (maxHops == null || maxHops <= 0 || maxHops > 10) {
                maxHops = 5;
            }

            Session session = getSession();
            try {
                // 修改：在Cypher查询中转换金额为浮点数
                String query = "MATCH path = shortestPath((a:Address {address: $fromAddress})" +
                        "-[r:TRANSFER*1.." + maxHops + "]->(b:Address {address: $toAddress})) " +
                        "WITH NODES(path) as nodes, RELATIONSHIPS(path) as rels, LENGTH(path) as hopCount " +
                        "RETURN " +
                        "  hopCount, " +
                        "  [node in nodes | node.address] as nodeAddresses, " +
                        "  [node in nodes | node.chain] as nodeChains, " +
                        "  [node in nodes | node.riskLevel] as nodeRiskLevels, " +
                        "  [rel in rels | rel.txHash] as relTxHashes, " +
                        "  [rel in rels | toFloat(rel.amount)] as relAmounts, " +  // 修改：转换为浮点数
                        "  [rel in rels | rel.time] as relTimes";

                Map<String, Object> params = new HashMap<>();
                params.put("fromAddress", fromAddress);
                params.put("toAddress", toAddress);

                Iterable<Map<String, Object>> queryResult = session.query(query, params, false);

                List<Map<String, Object>> results = new ArrayList<>();

                for (Map<String, Object> result : queryResult) {
                    Map<String, Object> pathInfo = new HashMap<>();
                    Integer hopCount = ((Number) result.get("hopCount")).intValue();
                    pathInfo.put("hopCount", hopCount);

                    // 使用辅助方法处理类型转换
                    List<String> nodeAddresses = safeConvertToList(result.get("nodeAddresses"), String.class);
                    List<String> nodeChains = safeConvertToList(result.get("nodeChains"), String.class);
                    List<Integer> nodeRiskLevels = safeConvertToList(result.get("nodeRiskLevels"), Integer.class);
                    List<String> relTxHashes = safeConvertToList(result.get("relTxHashes"), String.class);
                    List<Double> relAmounts = safeConvertToList(result.get("relAmounts"), Double.class);
                    List<String> relTimes = safeConvertToList(result.get("relTimes"), String.class);

                    // 构建节点列表
                    List<Map<String, Object>> nodes = new ArrayList<>();
                    for (int i = 0; i < nodeAddresses.size(); i++) {
                        Map<String, Object> nodeInfo = new HashMap<>();
                        nodeInfo.put("address", nodeAddresses.get(i));
                        nodeInfo.put("chain", i < nodeChains.size() ? nodeChains.get(i) : "");
                        nodeInfo.put("riskLevel", i < nodeRiskLevels.size() ? nodeRiskLevels.get(i) : 0);
                        nodes.add(nodeInfo);
                    }
                    pathInfo.put("nodes", nodes);

                    // 构建关系列表
                    List<Map<String, Object>> relations = new ArrayList<>();
                    for (int i = 0; i < relTxHashes.size(); i++) {
                        Map<String, Object> relInfo = new HashMap<>();
                        relInfo.put("txHash", relTxHashes.get(i));
                        relInfo.put("amount", i < relAmounts.size() ? relAmounts.get(i) : 0.0);
                        relInfo.put("time", i < relTimes.size() ? relTimes.get(i) : "");
                        relations.add(relInfo);
                    }
                    pathInfo.put("relations", relations);

                    results.add(pathInfo);
                }

                return ApiResponse.success(results, (long) results.size());

            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("查找交易路径失败: {} -> {}", fromAddress, toAddress, e);
            return ApiResponse.error(500, "查找交易路径失败: " + e.getMessage());
        }
    }

    // 添加通用的安全转换方法
    private <T> List<T> safeConvertToList(Object value, Class<T> targetType) {
        List<T> result = new ArrayList<>();

        if (value == null) {
            return result;
        }

        try {
            if (value instanceof List) {
                return (List<T>) value;
            } else if (value.getClass().isArray()) {
                if (targetType == String.class && value instanceof String[]) {
                    return (List<T>) Arrays.asList((String[]) value);
                } else if (targetType == Integer.class && value instanceof Integer[]) {
                    return (List<T>) Arrays.asList((Integer[]) value);
                } else if (targetType == Double.class && value instanceof Double[]) {
                    return (List<T>) Arrays.asList((Double[]) value);
                } else if (targetType == Double.class && value instanceof Float[]) {
                    // 处理Float数组转Double
                    Float[] floatArray = (Float[]) value;
                    for (Float f : floatArray) {
                        result.add((T) Double.valueOf(f));
                    }
                    return result;
                } else if (value instanceof Object[]) {
                    for (Object obj : (Object[]) value) {
                        result.add(convertValue(obj, targetType));
                    }
                    return result;
                }
            }

            // 其他情况尝试直接转换
            log.warn("无法直接转换的类型: {} to {}", value.getClass(), targetType);
        } catch (Exception e) {
            log.error("类型转换失败", e);
        }

        return result;
    }

    private <T> T convertValue(Object value, Class<T> targetType) {
        if (value == null) {
            if (targetType == Double.class) return (T) Double.valueOf(0.0);
            if (targetType == Integer.class) return (T) Integer.valueOf(0);
            if (targetType == String.class) return (T) "";
            return null;
        }

        try {
            if (targetType == Double.class) {
                if (value instanceof Number) {
                    return (T) Double.valueOf(((Number) value).doubleValue());
                } else if (value instanceof String) {
                    return (T) Double.valueOf(Double.parseDouble((String) value));
                }
            } else if (targetType == Integer.class) {
                if (value instanceof Number) {
                    return (T) Integer.valueOf(((Number) value).intValue());
                } else if (value instanceof String) {
                    return (T) Integer.valueOf(Integer.parseInt((String) value));
                }
            } else if (targetType == String.class) {
                return (T) value.toString();
            }
        } catch (Exception e) {
            log.error("值转换失败: {} to {}", value, targetType, e);
        }

        // 默认值
        if (targetType == Double.class) return (T) Double.valueOf(0.0);
        if (targetType == Integer.class) return (T) Integer.valueOf(0);
        if (targetType == String.class) return (T) "";

        return null;
    }

    // 添加辅助方法处理数组转换
    private List<String> convertToStringList(Object value) {
        List<String> result = new ArrayList<>();
        if (value == null) {
            return result;
        }

        if (value instanceof List) {
            return (List<String>) value;
        } else if (value.getClass().isArray()) {
            if (value instanceof String[]) {
                return Arrays.asList((String[]) value);
            } else if (value instanceof Object[]) {
                for (Object obj : (Object[]) value) {
                    result.add(obj != null ? obj.toString() : "");
                }
            }
        }
        return result;
    }

    private List<Integer> convertToIntegerList(Object value) {
        List<Integer> result = new ArrayList<>();
        if (value == null) {
            return result;
        }

        if (value instanceof List) {
            return (List<Integer>) value;
        } else if (value.getClass().isArray()) {
            if (value instanceof Integer[]) {
                return Arrays.asList((Integer[]) value);
            } else if (value instanceof int[]) {
                for (int num : (int[]) value) {
                    result.add(num);
                }
            } else if (value instanceof Number[]) {
                for (Number num : (Number[]) value) {
                    result.add(num != null ? num.intValue() : 0);
                }
            }
        }
        return result;
    }

    private List<Double> convertToDoubleList(Object value) {
        List<Double> result = new ArrayList<>();
        if (value == null) {
            return result;
        }

        if (value instanceof List) {
            return (List<Double>) value;
        } else if (value.getClass().isArray()) {
            if (value instanceof Double[]) {
                return Arrays.asList((Double[]) value);
            } else if (value instanceof double[]) {
                for (double num : (double[]) value) {
                    result.add(num);
                }
            } else if (value instanceof Number[]) {
                for (Number num : (Number[]) value) {
                    result.add(num != null ? num.doubleValue() : 0.0);
                }
            }
        }
        return result;
    }

    @Override
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
                String query = "MATCH (a:Address {address: $address})-[r:TRANSFER*1.." + maxHops + "]-(b:Address) " +
                        "WHERE a <> b " +
                        "WITH b, MIN(LENGTH(r)) as distance " +
                        "RETURN b.address as address, " +
                        "       b.balance as balance, " +
                        "       b.tx_count as txCount, " +
                        "       distance, " +
                        "       b.risk_score as riskScore " +
                        "ORDER BY distance, b.tx_count DESC";

                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                Iterable<Map<String, Object>> queryResult = session.query(query, params, false);

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

    @Override
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> getAddressTransferStats(String address) {
        try {
            Map<String, Object> stats = new HashMap<>();

            Session session = getSession();
            try {
                // 1. 使用Cypher查询地址基本信息，避免使用Repository
                String addressQuery = "MATCH (a:Address {address: $address}) " +
                        "RETURN a.address as address, a.chain as chain, " +
                        "a.firstSeen as firstSeen, a.lastSeen as lastSeen, " +
                        "a.riskLevel as riskLevel, a.tag as tag, " +
                        "a.balance as balance, a.txCount as txCount";

                Map<String, Object> params = Collections.singletonMap("address", address);
                Iterable<Map<String, Object>> addressResult = session.query(addressQuery, params, false);

                for (Map<String, Object> record : addressResult) {
                    stats.putAll(record);
                }

                // 2. 获取转账统计（使用toFloat转换金额）
                String sentQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN SUM(toFloat(r.amount)) as totalSent";

                Iterable<Map<String, Object>> sentResult = session.query(sentQuery, params, false);
                Double totalSent = 0.0;
                for (Map<String, Object> record : sentResult) {
                    if (record.get("totalSent") != null) {
                        totalSent = ((Number) record.get("totalSent")).doubleValue();
                    }
                }

                String receivedQuery = "MATCH (a:Address {address: $address})<-[r:TRANSFER]-(b:Address) " +
                        "RETURN SUM(toFloat(r.amount)) as totalReceived";

                Iterable<Map<String, Object>> receivedResult = session.query(receivedQuery, params, false);
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

    @Override
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> getTransferStatsBetweenAddresses(String fromAddress, String toAddress) {
        try {
            Map<String, Object> stats = new HashMap<>();

            Session session = getSession();
            try {
                String query = "MATCH (a:Address {address: $fromAddress})-[r:TRANSFER]->(b:Address {address: $toAddress}) " +
                        "RETURN COUNT(r) as transferCount, " +
                        "       SUM(toFloat(r.amount)) as totalAmount, " +
                        "       AVG(toFloat(r.amount)) as averageAmount, " +
                        "       MIN(r.time) as firstTransfer, " +
                        "       MAX(r.time) as lastTransfer";

                Map<String, Object> params = new HashMap<>();
                params.put("fromAddress", fromAddress);
                params.put("toAddress", toAddress);

                Iterable<Map<String, Object>> queryResult = session.query(query, params, false);

                for (Map<String, Object> record : queryResult) {
                    stats.put("fromAddress", fromAddress);
                    stats.put("toAddress", toAddress);
                    stats.put("transferCount", record.get("transferCount"));
                    stats.put("totalAmount", record.get("totalAmount"));
                    stats.put("averageAmount", record.get("averageAmount"));
                    stats.put("firstTransfer", record.get("firstTransfer"));
                    stats.put("lastTransfer", record.get("lastTransfer"));
                }

                if (stats.isEmpty()) {
                    stats.put("fromAddress", fromAddress);
                    stats.put("toAddress", toAddress);
                    stats.put("transferCount", 0);
                    stats.put("totalAmount", 0);
                    stats.put("averageAmount", 0);
                }

            } finally {
                closeSession(session);
            }

            return ApiResponse.success(stats, null);
        } catch (Exception e) {
            log.error("获取地址间转账统计失败: {} -> {}", fromAddress, toAddress, e);
            return ApiResponse.error(500, "获取地址间转账统计失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<List<Map<String, Object>>> findLargeTransfers(BigDecimal minAmount, LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        try {
            if (minAmount == null) {
                minAmount = BigDecimal.valueOf(100);
            }

            if (startTime == null) {
                startTime = LocalDateTime.now().minusDays(30);
            }

            if (endTime == null) {
                endTime = LocalDateTime.now();
            }

            if (limit == null || limit <= 0 || limit > 100) {
                limit = 50;
            }

            // 修复：Neo4j datetime函数需要ISO8601格式（带T分隔符）
            DateTimeFormatter isoFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss");
            String startTimeStr = startTime.format(isoFormatter);
            String endTimeStr = endTime.format(isoFormatter);

            Session session = getSession();
            try {
                String query = "MATCH (a:Address)-[r:TRANSFER]->(b:Address) " +
                        "WHERE toFloat(r.amount) >= $minAmount " +
                        "AND r.time >= datetime($startTime) " +
                        "AND r.time <= datetime($endTime) " +
                        "RETURN a.address as fromAddress, " +
                        "b.address as toAddress, " +
                        "toFloat(r.amount) as amount, " +
                        "r.txHash as txHash, " +
                        "r.time as time " +
                        "ORDER BY toFloat(r.amount) DESC " +
                        "LIMIT $limit";

                Map<String, Object> params = new HashMap<>();
                params.put("minAmount", minAmount.doubleValue());
                params.put("startTime", startTimeStr);  // 格式：2026-01-01T00:00:00
                params.put("endTime", endTimeStr);      // 格式：2026-01-31T23:59:59
                params.put("limit", limit);

                Iterable<Map<String, Object>> queryResult = session.query(query, params, false);

                List<Map<String, Object>> results = new ArrayList<>();
                for (Map<String, Object> record : queryResult) {
                    results.add(record);
                }

                return ApiResponse.success(results, (long) results.size());
            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("查找大额转账失败", e);
            return ApiResponse.error(500, "查找大额转账失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> analyzeAddressPattern(String address, Integer depth) {
        try {
            if (address == null || address.isEmpty()) {
                return ApiResponse.error(400, "地址参数不能为空");
            }

            if (depth == null || depth <= 0 || depth > 3) {
                depth = 2;
            }

            Map<String, Object> analysis = new HashMap<>();
            analysis.put("address", address);
            analysis.put("analysisDepth", depth);

            Session session = getSession();
            try {
                Map<String, Object> params = Collections.singletonMap("address", address);

                // 分析扇形模式（Fan-out） - 使用toFloat转换金额
                String fanOutQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN COUNT(DISTINCT b) as uniqueRecipients, " +
                        "SUM(toFloat(r.amount)) as totalSent, " +
                        "AVG(toFloat(r.amount)) as avgSent";

                Iterable<Map<String, Object>> fanOutResult = session.query(fanOutQuery, params, false);
                for (Map<String, Object> record : fanOutResult) {
                    analysis.put("fanOutPattern", record);
                }

                // 分析扇形模式（Fan-in）
                String fanInQuery = "MATCH (a:Address)<-[r:TRANSFER]-(b:Address {address: $address}) " +
                        "RETURN COUNT(DISTINCT a) as uniqueSenders, " +
                        "SUM(toFloat(r.amount)) as totalReceived, " +
                        "AVG(toFloat(r.amount)) as avgReceived";

                Iterable<Map<String, Object>> fanInResult = session.query(fanInQuery, params, false);
                for (Map<String, Object> record : fanInResult) {
                    analysis.put("fanInPattern", record);
                }

                // 分析交易频率
                String frequencyQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]-() " +
                        "RETURN COUNT(r) as totalTransfers, " +
                        "MIN(r.time) as firstTransfer, " +
                        "MAX(r.time) as lastTransfer";

                Iterable<Map<String, Object>> frequencyResult = session.query(frequencyQuery, params, false);
                for (Map<String, Object> record : frequencyResult) {
                    analysis.put("transferFrequency", record);
                }

                // 分析N跳内关联地址的统计
                String hopQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER*1.." + depth + "]-(b:Address) " +
                        "WITH b, MIN(LENGTH(r)) as distance " +
                        "RETURN distance, COUNT(DISTINCT b) as addressCount";

                Iterable<Map<String, Object>> hopResults = session.query(hopQuery, params, false);
                List<Map<String, Object>> hopStats = new ArrayList<>();
                for (Map<String, Object> hopResult : hopResults) {
                    hopStats.add(hopResult);
                }
                analysis.put("hopStatistics", hopStats);
            } finally {
                closeSession(session);
            }

            return ApiResponse.success(analysis, null);
        } catch (Exception e) {
            log.error("分析地址模式失败: {}", address, e);
            return ApiResponse.error(500, "分析地址模式失败: " + e.getMessage());
        }
    }

    @Override
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

                Iterable<Map<String, Object>> outgoingResult = session.query(outgoingQuery, params, false);
                List<String> outgoingConnections = new ArrayList<>();
                for (Map<String, Object> record : outgoingResult) {
                    outgoingConnections.add((String) record.get("address"));
                }

                // 获取入度连接
                String incomingQuery = "MATCH (a:Address)<-[r:TRANSFER]-(b:Address {address: $address}) " +
                        "RETURN DISTINCT a.address as address";
                Iterable<Map<String, Object>> incomingResult = session.query(incomingQuery, params, false);
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

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public ApiResponse<Void> updateAddressRiskLevel(String address, Integer riskLevel) {
        try {
            Session session = getSession();
            try {
                // 先检查地址是否存在
                String checkQuery = "MATCH (a:Address {address: $address}) RETURN a.address as address";
                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                Iterable<Map<String, Object>> checkResult = session.query(checkQuery, params, false);

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
                String updateQuery = "MATCH (a:Address {address: $address}) SET a.riskLevel = $riskLevel";
                params.put("riskLevel", riskLevel);

                session.query(updateQuery, params);

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

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public ApiResponse<Void> tagAddress(String address, String tag) {
        try {
            Session session = getSession();
            try {
                // 先检查地址是否存在
                String checkQuery = "MATCH (a:Address {address: $address}) RETURN a.address as address";
                Map<String, Object> params = new HashMap<>();
                params.put("address", address);

                Iterable<Map<String, Object>> checkResult = session.query(checkQuery, params, false);

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

                session.query(updateQuery, params);

                log.info("为地址打标签: {} -> {}", address, tag);
                return ApiResponse.success(null, null);
            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("为地址打标签失败: {}", address, e);
            return ApiResponse.error(500, "为地址打标签失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public void cleanGraphData(String chain) {
        try {
            if (chain == null || chain.isEmpty()) {
                chain = "ETH";
            }

            log.info("开始清理图数据库数据，链: {}", chain);

            Session session = getSession();
            try {
                String deleteQuery = "MATCH (n) WHERE n.chain = $chain DETACH DELETE n";
                session.query(deleteQuery, Collections.singletonMap("chain", chain));
            } finally {
                closeSession(session);
            }

            log.info("成功清理图数据库数据");
        } catch (Exception e) {
            log.error("清理图数据库失败", e);
            throw new RuntimeException("清理图数据库失败", e);
        }
    }

    @Override
    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> testNeo4jConnection() {
        try {
            Map<String, Object> result = new HashMap<>();

            Session session = getSession();
            try {
                // 方法1：使用更简单的查询
                String query = "RETURN 1 as test";
                org.neo4j.ogm.model.Result queryResult = session.query(query, Collections.emptyMap());

                if (queryResult != null) {
                    result.put("status", "connected");
                    result.put("message", "Neo4j连接成功");
                } else {
                    result.put("status", "error");
                    result.put("message", "查询返回空结果");
                }

                // 方法2：尝试查询节点数量（使用不同的方法）
                String countQuery = "MATCH (n) RETURN count(n) as nodeCount";
                Iterable<Map<String, Object>> countResult = session.query(countQuery, Collections.emptyMap(), false);

                Long nodeCount = 0L;
                for (Map<String, Object> row : countResult) {
                    if (row.get("nodeCount") != null) {
                        Object countObj = row.get("nodeCount");
                        if (countObj instanceof Number) {
                            nodeCount = ((Number) countObj).longValue();
                        } else if (countObj instanceof Long) {
                            nodeCount = (Long) countObj;
                        } else if (countObj instanceof Integer) {
                            nodeCount = ((Integer) countObj).longValue();
                        }
                    }
                }

                result.put("nodeCount", nodeCount);
                result.put("timestamp", LocalDateTime.now().toString());

            } catch (Exception e) {
                log.error("测试Neo4j连接失败", e);
                result.put("status", "error");
                result.put("message", "连接失败: " + e.getMessage());
            } finally {
                closeSession(session);
            }

            // 如果连接成功，返回成功响应
            if ("connected".equals(result.get("status"))) {
                return ApiResponse.success(result, null);
            } else {
                return ApiResponse.error(500, "Neo4j连接测试失败: " + result.get("message"));
            }
        } catch (Exception e) {
            log.error("测试Neo4j连接失败", e);
            return ApiResponse.error(500, "测试Neo4j连接失败: " + e.getMessage());
        }
    }

    // 异步保存图数据
    public CompletableFuture<Void> saveTransactionToGraphAsync(ChainTx chainTx) {
        return CompletableFuture.runAsync(() -> {
            try {
                saveTransactionToGraph(chainTx);
            } catch (Exception e) {
                log.error("异步保存图数据失败: {}", chainTx.getTxHash(), e);
            }
        });
    }
}