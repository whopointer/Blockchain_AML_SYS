package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.po.ChainTx;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.session.Session;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class GraphTransactionService extends AbstractGraphService {

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
                    .append("ON CREATE SET a1.first_seen = $blockTime, a1.last_seen = $blockTime, ")
                    .append("a1.risk_level = 0, a1.tag = '' ")
                    .append("ON MATCH SET a1.last_seen = $blockTime ");

            // 创建/更新接收地址（如果存在）
            if (chainTx.getToAddress() != null && !chainTx.getToAddress().isEmpty()) {
                params.put("toAddress", chainTx.getToAddress());

                cypher.append("MERGE (a2:Address {address: $toAddress, chain: $chain}) ")
                        .append("ON CREATE SET a2.first_seen = $blockTime, a2.last_seen = $blockTime, ")
                    .append("a2.risk_level = 0, a2.tag = '' ")
                    .append("ON MATCH SET a2.last_seen = $blockTime ");
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

    // 简化版Cypher保存方法，避免复杂的关系映射
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
            cypher.append("ON CREATE SET from.first_seen = $blockTime, from.last_seen = $blockTime, ");
            cypher.append("from.risk_level = 0, from.tag = '' ");
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
                cypher.append("ON CREATE SET to.first_seen = $blockTime, to.last_seen = $blockTime, ");
                cypher.append("to.risk_level = 0, to.tag = '' ");
                cypher.append("ON MATCH SET to.lastSeen = $blockTime ");

                // 创建TRANSFER关系
                cypher.append("WITH from, to, tx ");
                cypher.append("MERGE (from)-[tr:TRANSFER]->(to) ");
                cypher.append("ON CREATE SET tr.txHash = $txHash, tr.amount = $totalOutput, tr.time = $blockTime ");
                cypher.append("ON MATCH SET tr.amount = $totalOutput, tr.time = $blockTime ");
                // 由于可能存在多条交易记录，每次MATCH时也更新txHash，确保最新交易的哈希被保存
                cypher.append("SET tr.txHash = $txHash ");

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
}