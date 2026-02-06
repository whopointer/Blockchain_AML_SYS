package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.session.Session;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class GraphStatsService extends AbstractGraphService {

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
                        "toFloat(COALESCE(r.amount, 0)) as amount, " +
                        "CASE WHEN 'txHash' IN keys(r) THEN toString(r['txHash']) ELSE '' END as txHash, " +
                        "CASE WHEN 'time' IN keys(r) THEN toString(r['time']) ELSE '' END as time " +
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
                String hopQuery = "MATCH path = (a:Address {address: $address})-[r:TRANSFER*1.." + depth + "]-(b:Address) " +
                        "WITH b, path, LENGTH(path) as distance " +
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

    @Transactional(readOnly = true, transactionManager = "neo4jTransactionManager")
    public ApiResponse<Map<String, Object>> testNeo4jConnection() {
        try {
            Map<String, Object> result = new HashMap<>();
            Session session = getSession();
            try {
                // 简单查询验证连接
                String query = "MATCH (a:Address) RETURN COUNT(a) as addressCount LIMIT 1";
                Iterable<Map<String, Object>> queryResult = session.query(query, Collections.emptyMap(), false);
                
                for (Map<String, Object> record : queryResult) {
                    result.put("addressCount", record.get("addressCount"));
                    result.put("connectionStatus", "SUCCESS");
                    return ApiResponse.success(result, null);
                }
                
                result.put("connectionStatus", "SUCCESS");
                result.put("addressCount", 0);
                return ApiResponse.success(result, null);
            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("Neo4j连接测试失败", e);
            return ApiResponse.error(500, "Neo4j连接失败: " + e.getMessage());
        }
    }

    @Transactional(transactionManager = "neo4jTransactionManager")
    public void cleanGraphData(String chain) {
        try {
            Session session = getSession();
            try {
                // 删除指定链的数据
                String deleteQuery = "MATCH (n) WHERE n.chain = $chain DETACH DELETE n";
                Map<String, Object> params = Collections.singletonMap("chain", chain);
                session.query(deleteQuery, params);
                log.info("清理链 {} 的图数据成功", chain);
            } finally {
                closeSession(session);
            }
        } catch (Exception e) {
            log.error("清理图数据失败: {}", chain, e);
            throw e;
        }
    }
}