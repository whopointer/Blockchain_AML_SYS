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

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveTransactionToGraph(ChainTx chainTx) {
        Session session = getSession();
        try {
            // 1. 创建或更新发送地址节点
            AddressNode fromNode = getOrCreateAddressNode(chainTx.getFromAddress(), chainTx.getChain(), chainTx.getBlockTime());

            // 2. 创建或更新接收地址节点
            AddressNode toNode = getOrCreateAddressNode(chainTx.getToAddress(), chainTx.getChain(), chainTx.getBlockTime());

            // 3. 创建交易节点
            TransactionNode txNode = getOrCreateTransactionNode(chainTx);

            // 4. 创建地址到地址的转账关系（第一种关系）
            createTransferRelation(fromNode, toNode, chainTx);

            // 5. 创建地址到交易的关系（第二种更细粒度的关系）
            Integer txIndex = getTxIndex(chainTx);
            createSpentRelation(fromNode, txNode, chainTx, txIndex);
            createOutputRelation(txNode, toNode, chainTx, txIndex);

            // 保存节点和关系
            session.save(fromNode);
            session.save(toNode);
            session.save(txNode);

            log.debug("成功保存交易到图数据库: {}", chainTx.getTxHash());
        } catch (Exception e) {
            log.error("保存交易到图数据库失败: {}", chainTx.getTxHash(), e);
            throw new RuntimeException("保存图数据失败", e);
        } finally {
            closeSession(session);
        }
    }

    private Integer getTxIndex(ChainTx chainTx) {
        if (chainTx.getTxIndex() != null) {
            return chainTx.getTxIndex();
        }
        return Math.abs(chainTx.getTxHash().hashCode() % 1000);
    }

    private AddressNode getOrCreateAddressNode(String address, String chain, LocalDateTime timestamp) {
        if (address == null || address.isEmpty()) {
            return null;
        }

        AddressNode node = addressNodeRepository.findByAddressAndChain(address, chain);
        if (node == null) {
            node = new AddressNode(chain, address);
            node.setFirstSeen(timestamp);
            node.setLastSeen(timestamp);
            node.setRiskLevel(0);
            node.setTag("");
        } else {
            node.setLastSeen(timestamp);
        }

        return node;
    }

    private TransactionNode getOrCreateTransactionNode(ChainTx chainTx) {
        TransactionNode node = transactionNodeRepository.findByTxHashAndChain(chainTx.getTxHash(), chainTx.getChain());
        if (node == null) {
            node = new TransactionNode(chainTx.getChain(), chainTx.getTxHash());
            node.setBlockHeight(chainTx.getBlockHeight());
            node.setTime(chainTx.getBlockTime());
            node.setTotalInput(chainTx.getTotalInput());
            node.setTotalOutput(chainTx.getTotalOutput());
            node.setFee(chainTx.getFee());
        }

        return node;
    }

    private void createTransferRelation(AddressNode fromNode, AddressNode toNode, ChainTx chainTx) {
        if (fromNode != null && toNode != null && chainTx.getTotalOutput() != null) {
            TransferRelation relation = new TransferRelation();
            relation.setFromAddress(fromNode);
            relation.setToAddress(toNode);
            relation.setTxHash(chainTx.getTxHash());
            relation.setAmount(chainTx.getTotalOutput());
            relation.setTime(chainTx.getBlockTime());

            fromNode.getOutgoingTransfers().add(relation);
            toNode.getIncomingTransfers().add(relation);
        }
    }

    private void createSpentRelation(AddressNode fromNode, TransactionNode txNode, ChainTx chainTx, Integer index) {
        if (fromNode != null && chainTx.getTotalInput() != null) {
            SpentRelation relation = new SpentRelation();
            relation.setFromAddress(fromNode);
            relation.setTransaction(txNode);
            relation.setAmount(chainTx.getTotalInput());
            relation.setIndex(index);

            fromNode.getSpentTransactions().add(relation);
            txNode.getFromAddresses().add(relation);
        }
    }

    private void createOutputRelation(TransactionNode txNode, AddressNode toNode, ChainTx chainTx, Integer index) {
        if (toNode != null && chainTx.getTotalOutput() != null) {
            OutputRelation relation = new OutputRelation();
            relation.setTransaction(txNode);
            relation.setToAddress(toNode);
            relation.setAmount(chainTx.getTotalOutput());
            relation.setIndex(index);

            txNode.getToAddresses().add(relation);
            toNode.getReceivedTransactions().add(relation);
        }
    }

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveTransactionsToGraph(List<ChainTx> chainTxs) {
        log.info("开始批量保存 {} 笔交易到图数据库", chainTxs.size());

        int successCount = 0;
        int failCount = 0;

        Session session = getSession();
        try {
            for (ChainTx tx : chainTxs) {
                try {
                    saveTransactionToGraph(tx);
                    successCount++;

                    if (successCount % 100 == 0) {
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

        log.info("批量保存完成: 成功 {} 笔, 失败 {} 笔", successCount, failCount);
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

            List<Map<String, Object>> results = new ArrayList<>();

            Session session = getSession();
            try {
                String query = "MATCH path = shortestPath((a:Address {address: $fromAddress})" +
                        "-[:TRANSFER*1.." + maxHops + "]->(b:Address {address: $toAddress})) " +
                        "RETURN NODES(path) as nodes, RELATIONSHIPS(path) as rels, LENGTH(path) as hopCount";

                Map<String, Object> params = new HashMap<>();
                params.put("fromAddress", fromAddress);
                params.put("toAddress", toAddress);

                // 修复：添加类型转换
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> queryResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, query, params);

                for (Map<String, Object> result : queryResult) {
                    Map<String, Object> pathInfo = new HashMap<>();
                    pathInfo.put("hopCount", result.get("hopCount"));

                    // 处理节点
                    List<Map<String, Object>> nodes = new ArrayList<>();
                    @SuppressWarnings("unchecked")
                    Iterable<Map<String, Object>> addressNodes = (Iterable<Map<String, Object>>) result.get("nodes");
                    for (Map<String, Object> node : addressNodes) {
                        Map<String, Object> nodeInfo = new HashMap<>();
                        nodeInfo.put("address", node.get("address"));
                        nodeInfo.put("chain", node.get("chain"));
                        nodeInfo.put("riskLevel", node.get("risk_level"));
                        nodes.add(nodeInfo);
                    }
                    pathInfo.put("nodes", nodes);

                    // 处理关系
                    List<Map<String, Object>> relations = new ArrayList<>();
                    @SuppressWarnings("unchecked")
                    Iterable<Map<String, Object>> transferRels = (Iterable<Map<String, Object>>) result.get("rels");
                    for (Map<String, Object> rel : transferRels) {
                        Map<String, Object> relInfo = new HashMap<>();
                        relInfo.put("txHash", rel.get("tx_hash"));
                        relInfo.put("amount", rel.get("amount"));
                        relInfo.put("time", rel.get("time"));
                        relations.add(relInfo);
                    }
                    pathInfo.put("relations", relations);

                    results.add(pathInfo);
                }
            } finally {
                closeSession(session);
            }

            return ApiResponse.success(results, (long) results.size());
        } catch (Exception e) {
            log.error("查找交易路径失败: {} -> {}", fromAddress, toAddress, e);
            return ApiResponse.error(500, "查找交易路径失败: " + e.getMessage());
        }
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
                String query = "MATCH (a:Address {address: $address})-[r:TRANSFER*1..$maxHops]-(b:Address) " +
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
                params.put("maxHops", maxHops);

                // 修复：添加类型转换
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> queryResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, query, params);

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

            // 获取基本地址信息
            AddressNode addressNode = addressNodeRepository.findByAddress(address);
            if (addressNode != null) {
                stats.put("address", addressNode.getAddress());
                stats.put("chain", addressNode.getChain());
                stats.put("firstSeen", addressNode.getFirstSeen());
                stats.put("lastSeen", addressNode.getLastSeen());
                stats.put("riskLevel", addressNode.getRiskLevel());
                stats.put("tag", addressNode.getTag());
            }

            Session session = getSession();
            try {
                // 获取转账统计
                String sentQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN SUM(r.amount) as totalSent";
                Map<String, Object> sentParams = Collections.singletonMap("address", address);
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> sentResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, sentQuery, sentParams);

                Double totalSent = 0.0;
                for (Map<String, Object> record : sentResult) {
                    if (record.get("totalSent") != null) {
                        totalSent = ((Number) record.get("totalSent")).doubleValue();
                    }
                }

                String receivedQuery = "MATCH (a:Address {address: $address})<-[r:TRANSFER]-(b:Address) " +
                        "RETURN SUM(r.amount) as totalReceived";
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> receivedResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, receivedQuery, sentParams);

                Double totalReceived = 0.0;
                for (Map<String, Object> record : receivedResult) {
                    if (record.get("totalReceived") != null) {
                        totalReceived = ((Number) record.get("totalReceived")).doubleValue();
                    }
                }

                stats.put("totalSent", totalSent);
                stats.put("totalReceived", totalReceived);

                // 获取直接关联地址
                String toQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN DISTINCT b.address as toAddress";
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> toResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, toQuery, sentParams);

                List<String> transferTo = new ArrayList<>();
                for (Map<String, Object> record : toResult) {
                    transferTo.add((String) record.get("toAddress"));
                }

                String fromQuery = "MATCH (a:Address {address: $address})<-[r:TRANSFER]-(b:Address) " +
                        "RETURN DISTINCT b.address as fromAddress";
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> fromResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, fromQuery, sentParams);

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
                        "       SUM(r.amount) as totalAmount, " +
                        "       AVG(r.amount) as averageAmount, " +
                        "       MIN(r.time) as firstTransfer, " +
                        "       MAX(r.time) as lastTransfer";

                Map<String, Object> params = new HashMap<>();
                params.put("fromAddress", fromAddress);
                params.put("toAddress", toAddress);

                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> queryResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, query, params);

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

            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String startTimeStr = startTime.format(formatter);
            String endTimeStr = endTime.format(formatter);

            Session session = getSession();
            try {
                String query = "MATCH (a:Address)-[r:TRANSFER]->(b:Address) " +
                        "WHERE r.amount >= $minAmount " +
                        "AND r.time >= datetime($startTime) " +
                        "AND r.time <= datetime($endTime) " +
                        "RETURN a.address as fromAddress, " +
                        "b.address as toAddress, " +
                        "r.amount as amount, " +
                        "r.tx_hash as txHash, " +
                        "r.time as time " +
                        "ORDER BY r.amount DESC " +
                        "LIMIT $limit";

                Map<String, Object> params = new HashMap<>();
                params.put("minAmount", minAmount.doubleValue());
                params.put("startTime", startTimeStr);
                params.put("endTime", endTimeStr);
                params.put("limit", limit);

                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> queryResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, query, params);

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
                // 分析扇形模式（Fan-out）
                String fanOutQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
                        "RETURN COUNT(DISTINCT b) as uniqueRecipients, " +
                        "SUM(r.amount) as totalSent, " +
                        "AVG(r.amount) as avgSent";

                Map<String, Object> params = Collections.singletonMap("address", address);
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> fanOutResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, fanOutQuery, params);

                for (Map<String, Object> record : fanOutResult) {
                    analysis.put("fanOutPattern", record);
                }

                // 分析扇形模式（Fan-in）
                String fanInQuery = "MATCH (a:Address)<-[r:TRANSFER]-(b:Address {address: $address}) " +
                        "RETURN COUNT(DISTINCT a) as uniqueSenders, " +
                        "SUM(r.amount) as totalReceived, " +
                        "AVG(r.amount) as avgReceived";

                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> fanInResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, fanInQuery, params);
                for (Map<String, Object> record : fanInResult) {
                    analysis.put("fanInPattern", record);
                }

                // 分析交易频率
                String frequencyQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER]-() " +
                        "RETURN COUNT(r) as totalTransfers, " +
                        "MIN(r.time) as firstTransfer, " +
                        "MAX(r.time) as lastTransfer";

                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> frequencyResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, frequencyQuery, params);
                for (Map<String, Object> record : frequencyResult) {
                    analysis.put("transferFrequency", record);
                }

                // 分析N跳内关联地址的统计
                String hopQuery = "MATCH (a:Address {address: $address})-[r:TRANSFER*1.." + depth + "]-(b:Address) " +
                        "WITH b, MIN(LENGTH(r)) as distance " +
                        "RETURN distance, COUNT(DISTINCT b) as addressCount";

                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> hopResults = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, hopQuery, params);

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

                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> outgoingResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, outgoingQuery, params);
                List<String> outgoingConnections = new ArrayList<>();
                for (Map<String, Object> record : outgoingResult) {
                    outgoingConnections.add((String) record.get("address"));
                }

                // 获取入度连接
                String incomingQuery = "MATCH (a:Address)<-[r:TRANSFER]-(b:Address {address: $address}) " +
                        "RETURN DISTINCT a.address as address";
                @SuppressWarnings("unchecked")
                Iterable<Map<String, Object>> incomingResult = (Iterable<Map<String, Object>>) (Object) session.query(Map.class, incomingQuery, params);
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
            AddressNode addressNode = addressNodeRepository.findByAddress(address);
            if (addressNode == null) {
                return ApiResponse.error(404, "地址不存在: " + address);
            }

            addressNode.setRiskLevel(riskLevel);

            Session session = getSession();
            try {
                session.save(addressNode);
            } finally {
                closeSession(session);
            }

            log.info("更新地址风险等级: {} -> {}", address, riskLevel);
            return ApiResponse.success(null, null);
        } catch (Exception e) {
            log.error("更新地址风险等级失败: {}", address, e);
            return ApiResponse.error(500, "更新地址风险等级失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional(transactionManager = "neo4jTransactionManager")
    public ApiResponse<Void> tagAddress(String address, String tag) {
        try {
            AddressNode addressNode = addressNodeRepository.findByAddress(address);
            if (addressNode == null) {
                return ApiResponse.error(404, "地址不存在: " + address);
            }

            addressNode.setTag(tag);

            Session session = getSession();
            try {
                session.save(addressNode);
            } finally {
                closeSession(session);
            }

            log.info("为地址打标签: {} -> {}", address, tag);
            return ApiResponse.success(null, null);
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