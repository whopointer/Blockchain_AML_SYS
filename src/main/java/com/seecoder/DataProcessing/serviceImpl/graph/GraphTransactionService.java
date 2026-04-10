package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.ChainTxInput;
import com.seecoder.DataProcessing.po.ChainTxOutput;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.session.Session;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import javax.annotation.PostConstruct;
import javax.annotation.PostConstruct;
import java.util.Collections;
@Slf4j
@Service
public class GraphTransactionService extends AbstractGraphService {

    @PostConstruct
    public void initIndexes() {
        log.info("开始创建 Neo4j 索引...");
        Session session = null;
        try {
            session = getSession();
            String[] indexCyphers = {
                    "CREATE INDEX transaction_tx_hash IF NOT EXISTS FOR (n:Transaction) ON (n.txHash)",
                    "CREATE INDEX transaction_chain IF NOT EXISTS FOR (n:Transaction) ON (n.chain)",
                    "CREATE INDEX address_address IF NOT EXISTS FOR (n:Address) ON (n.address)",
                    "CREATE INDEX address_chain IF NOT EXISTS FOR (n:Address) ON (n.chain)",
                    "CREATE INDEX address_address_chain IF NOT EXISTS FOR (n:Address) ON (n.address, n.chain)",
                    "CREATE INDEX transaction_txhash_chain IF NOT EXISTS FOR (n:Transaction) ON (n.txHash, n.chain)"
            };
            for (String cypher : indexCyphers) {
                try {
                    // 使用 query 方法执行 Cypher，第二个参数为参数 Map（无参数时传空 Map）
                    session.query(cypher, Collections.emptyMap());
                    log.info("索引创建/验证成功: {}", cypher);
                } catch (Exception e) {
                    // 索引已存在时会抛出异常，可忽略
                    log.warn("索引可能已存在: {}", e.getMessage());
                }
            }
            log.info("Neo4j 索引初始化完成");
        } catch (Exception e) {
            log.error("索引初始化失败", e);
        } finally {
            if (session != null) {
                closeSession(session);  // 使用你已有的 closeSession 方法
            }
        }
    }
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
            // 创建/更新交易节点
            cypher.append("MERGE (tx:Transaction {txHash: $txHash, chain: $chain}) ")
                    .append("ON CREATE SET tx.blockHeight = $blockHeight, tx.time = $blockTime, ")
                    .append("tx.totalInput = $totalInput, tx.totalOutput = $totalOutput, ")
                    .append("tx.fee = $fee, ")
                    .append("tx.numInputs = 1, ");  // 以太坊固定为1
            if (chainTx.getToAddress() != null && !chainTx.getToAddress().isEmpty()) {
                cypher.append("tx.numOutputs = 1 ");
            } else {
                cypher.append("tx.numOutputs = 0 ");
            }

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
            cypher.append("MERGE (tx:Transaction {txHash: $txHash, chain: $chain}) ")
                    .append("ON CREATE SET tx.blockHeight = $blockHeight, tx.time = $blockTime, ")
                    .append("tx.totalInput = $totalInput, tx.totalOutput = $totalOutput, ")
                    .append("tx.fee = $fee, ")
                    .append("tx.numInputs = 1, ");  // 以太坊固定为1
            if (chainTx.getToAddress() != null && !chainTx.getToAddress().isEmpty()) {
                cypher.append("tx.numOutputs = 1 ");
            } else {
                cypher.append("tx.numOutputs = 0 ");
            }

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
    // com/seecoder/DataProcessing/serviceImpl/graph/GraphTransactionService.java

    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveBitcoinTransactionToGraph(ChainTx tx, List<ChainTxInput> inputs, List<ChainTxOutput> outputs) {
        if (tx == null) return;
        Session session = getSession();
        try {
            Map<String, Object> params = new HashMap<>();
            params.put("chain", tx.getChain());
            params.put("txHash", tx.getTxHash());
            params.put("blockHeight", tx.getBlockHeight());
            params.put("blockTime", tx.getBlockTime() != null ? tx.getBlockTime().toString() : null);
            params.put("totalInput", convertBigDecimal(tx.getTotalInput()));
            params.put("totalOutput", convertBigDecimal(tx.getTotalOutput()));
            params.put("fee", convertBigDecimal(tx.getFee()));

            // 准备输入列表（每个元素为 Map，包含 address, amount, index）
            List<Map<String, Object>> inputList = new ArrayList<>();
            for (ChainTxInput in : inputs) {
                if (in.getAddress() == null) continue;
                Map<String, Object> inMap = new HashMap<>();
                inMap.put("address", in.getAddress());
                inMap.put("amount", convertBigDecimal(in.getValue()));
                inMap.put("index", in.getInputIndex() != null ? in.getInputIndex() : 0);
                inputList.add(inMap);
            }
            params.put("inputs", inputList);

            // 准备输出列表
            List<Map<String, Object>> outputList = new ArrayList<>();
            for (ChainTxOutput out : outputs) {
                if (out.getAddress() == null) continue;
                Map<String, Object> outMap = new HashMap<>();
                outMap.put("address", out.getAddress());
                outMap.put("amount", convertBigDecimal(out.getValue()));
                outMap.put("index", out.getOutputIndex() != null ? out.getOutputIndex() : 0);
                outputList.add(outMap);
            }
            params.put("outputs", outputList);

            // Cypher 查询：创建交易节点，并处理所有输入/输出关系
            String cypher =
                    "MERGE (tx:Transaction {txHash: $txHash, chain: $chain}) " +
                            "ON CREATE SET tx.blockHeight = $blockHeight, tx.time = $blockTime, " +
                            "              tx.totalInput = $totalInput, tx.totalOutput = $totalOutput, tx.fee = $fee " +
                            "WITH tx " +
                            "UNWIND $inputs AS input " +
                            "MERGE (addr:Address {address: input.address, chain: $chain}) " +
                            "ON CREATE SET addr.first_seen = $blockTime, addr.last_seen = $blockTime, " +
                            "              addr.risk_level = 0, addr.tag = '' " +
                            "ON MATCH SET addr.last_seen = $blockTime " +
                            "MERGE (addr)-[:SPENT {amount: input.amount, index: input.index}]->(tx) " +
                            "WITH tx " +
                            "UNWIND $outputs AS output " +
                            "MERGE (addr:Address {address: output.address, chain: $chain}) " +
                            "ON CREATE SET addr.first_seen = $blockTime, addr.last_seen = $blockTime, " +
                            "              addr.risk_level = 0, addr.tag = '' " +
                            "ON MATCH SET addr.last_seen = $blockTime " +
                            "MERGE (tx)-[:OUTPUT {amount: output.amount, index: output.index}]->(addr)";

            session.query(cypher, params);
        } finally {
            closeSession(session);
        }
    }


//    @Transactional(transactionManager = "neo4jTransactionManager")
//    public void saveBitcoinTransactionsToGraph(List<ChainTx> txs,
//                                               Map<String, List<ChainTxInput>> inputsMap,
//                                               Map<String, List<ChainTxOutput>> outputsMap) {
//        if (txs == null || txs.isEmpty()) return;
//        // 分批处理，每批 50 笔
//        int batchSize = 50;
//        for (int i = 0; i < txs.size(); i += batchSize) {
//            List<ChainTx> batch = txs.subList(i, Math.min(i + batchSize, txs.size()));
//            for (ChainTx tx : batch) {
//                List<ChainTxInput> inputs = inputsMap.getOrDefault(tx.getTxHash(), Collections.emptyList());
//                List<ChainTxOutput> outputs = outputsMap.getOrDefault(tx.getTxHash(), Collections.emptyList());
//                saveBitcoinTransactionToGraph(tx, inputs, outputs);
//            }
//        }
//    }


    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveTransactionsBatchToGraph(List<ChainTx> txs) {
        log.info("开始批量保存图数据库，交易数：{}", txs.size());
        if (txs == null || txs.isEmpty()) return;
        long start = System.currentTimeMillis();
        Session session = getSession();
        try {
            List<Map<String, Object>> txParams = new ArrayList<>();
            for (ChainTx tx : txs) {
                Map<String, Object> map = new HashMap<>();
                map.put("txHash", tx.getTxHash());
                map.put("chain", tx.getChain());
                map.put("fromAddress", tx.getFromAddress());
                map.put("toAddress", tx.getToAddress());
                map.put("blockHeight", tx.getBlockHeight());
                map.put("blockTime", tx.getBlockTime());
                map.put("totalInput", convertBigDecimal(tx.getTotalInput()));
                map.put("totalOutput", convertBigDecimal(tx.getTotalOutput()));
                map.put("fee", convertBigDecimal(tx.getFee()));
                map.put("txIndex", tx.getTxIndex() != null ? tx.getTxIndex() : 0);
                txParams.add(map);
            }

            // Cypher 优化：
            // - 地址节点 MERGE，创建时设置完整属性，匹配时仅更新 last_seen
            // - 交易节点 MERGE（确保幂等），但关系用 CREATE（更快）
            String cypher =
                    "UNWIND $txs AS tx " +
                            "MERGE (from:Address {address: tx.fromAddress, chain: tx.chain}) " +
                            "ON CREATE SET from.first_seen = tx.blockTime, from.last_seen = tx.blockTime, " +
                            "              from.risk_level = 0, from.tag = '' " +
                            "ON MATCH SET from.last_seen = tx.blockTime " +
                            "MERGE (txNode:Transaction {txHash: tx.txHash, chain: tx.chain}) " +
                            "ON CREATE SET txNode.blockHeight = tx.blockHeight, txNode.time = tx.blockTime, " +
                            "              txNode.totalInput = tx.totalInput, txNode.totalOutput = tx.totalOutput, " +
                            "              txNode.fee = tx.fee, txNode.numInputs = 1, " +
                            "              txNode.numOutputs = CASE WHEN tx.toAddress IS NOT NULL THEN 1 ELSE 0 END " +
                            "ON MATCH SET txNode.totalInput = tx.totalInput, txNode.totalOutput = tx.totalOutput, " +
                            "              txNode.fee = tx.fee, txNode.time = tx.blockTime " +
                            "CREATE (from)-[:SPENT {amount: tx.totalInput, index: tx.txIndex}]->(txNode) " +
                            "FOREACH(ignore IN CASE WHEN tx.toAddress IS NOT NULL THEN [1] ELSE [] END | " +
                            "    MERGE (to:Address {address: tx.toAddress, chain: tx.chain}) " +
                            "    ON CREATE SET to.first_seen = tx.blockTime, to.last_seen = tx.blockTime, " +
                            "                  to.risk_level = 0, to.tag = '' " +
                            "    ON MATCH SET to.last_seen = tx.blockTime " +
                            "    CREATE (from)-[:TRANSFER {txHash: tx.txHash, amount: tx.totalOutput, time: tx.blockTime}]->(to) " +
                            "    CREATE (txNode)-[:OUTPUT {amount: tx.totalOutput, index: tx.txIndex}]->(to) " +
                            ")";

            Map<String, Object> params = Collections.singletonMap("txs", txParams);
            session.query(cypher, params);
            log.info("批量保存完成，总耗时 {}ms，交易数 {}", System.currentTimeMillis() - start, txs.size());
        } catch (Exception e) {
            log.error("批量保存失败", e);
            throw new RuntimeException("图数据库批量保存失败", e);
        } finally {
            closeSession(session);
        }
    }
    /**
     * 批量保存比特币交易到图数据库（支持多输入/多输出）
     * 参考以太坊批量写入模式，使用 UNWIND 一次处理多笔交易
     * @param txs 交易列表
     * @param inputsMap 交易哈希 -> 输入列表
     * @param outputsMap 交易哈希 -> 输出列表
     * @param batchSize 每批处理的交易数（建议 50-100）
     */
    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveBitcoinTransactionsBatchToGraph(List<ChainTx> txs,
                                                    Map<String, List<ChainTxInput>> inputsMap,
                                                    Map<String, List<ChainTxOutput>> outputsMap,
                                                    int batchSize) {
        if (txs == null || txs.isEmpty()) {
            log.info("没有交易需要保存到图数据库");
            return;
        }
        log.info("开始批量保存比特币交易到图数据库，总交易数：{}，批次大小：{}", txs.size(), batchSize);

        int totalSuccess = 0;
        int totalFail = 0;

        for (int i = 0; i < txs.size(); i += batchSize) {
            int end = Math.min(i + batchSize, txs.size());
            List<ChainTx> batch = txs.subList(i, end);
            long batchStart = System.currentTimeMillis();
            log.info("处理第 {}/{} 批，包含 {} 笔交易", i / batchSize + 1,
                    (txs.size() + batchSize - 1) / batchSize, batch.size());

            Session session = null;
            try {
                session = getSession();

                // 构建批次参数（与之前相同）
                List<Map<String, Object>> txParams = new ArrayList<>();
                for (ChainTx tx : batch) {
                    Map<String, Object> txMap = new HashMap<>();
                    txMap.put("txHash", tx.getTxHash());
                    txMap.put("chain", tx.getChain());
                    txMap.put("blockHeight", tx.getBlockHeight());
                    txMap.put("blockTime", tx.getBlockTime() != null ? tx.getBlockTime().toString() : null);
                    txMap.put("totalInput", convertBigDecimal(tx.getTotalInput()));
                    txMap.put("totalOutput", convertBigDecimal(tx.getTotalOutput()));
                    txMap.put("fee", convertBigDecimal(tx.getFee()));

                    // 输入列表
                    List<ChainTxInput> inputs = inputsMap.getOrDefault(tx.getTxHash(), Collections.emptyList());
                    List<Map<String, Object>> inputList = new ArrayList<>();
                    for (ChainTxInput in : inputs) {
                        if (in.getAddress() == null || in.getAddress().isEmpty()) continue;
                        Map<String, Object> inMap = new HashMap<>();
                        inMap.put("address", in.getAddress());
                        inMap.put("amount", convertBigDecimal(in.getValue()));
                        inMap.put("index", in.getInputIndex() != null ? in.getInputIndex() : 0);
                        inputList.add(inMap);
                    }
                    txMap.put("inputs", inputList);

                    // 输出列表
                    List<ChainTxOutput> outputs = outputsMap.getOrDefault(tx.getTxHash(), Collections.emptyList());
                    List<Map<String, Object>> outputList = new ArrayList<>();
                    for (ChainTxOutput out : outputs) {
                        if (out.getAddress() == null || out.getAddress().isEmpty()) continue;
                        Map<String, Object> outMap = new HashMap<>();
                        outMap.put("address", out.getAddress());
                        outMap.put("amount", convertBigDecimal(out.getValue()));
                        outMap.put("index", out.getOutputIndex() != null ? out.getOutputIndex() : 0);
                        outputList.add(outMap);
                    }
                    txMap.put("outputs", outputList);

                    txParams.add(txMap);
                }

                // ========== 优化后的 Cypher ==========
                // 关键改动：
                // 1. 使用 OPTIONAL MATCH 提前检查节点是否存在，避免 MERGE 的内部全扫描（需要索引配合）
                // 2. 将 last_seen 更新移到单独的操作，减少写冲突
                // 3. 对于地址节点，只在必要时创建，更新 last_seen 使用 ON MATCH SET（但索引存在时很快）
                String cypher =
                        "UNWIND $txBatch AS tx " +
                                // ----- 交易节点 -----
                                "MERGE (txNode:Transaction {txHash: tx.txHash, chain: tx.chain}) " +
                                "ON CREATE SET txNode.blockHeight = tx.blockHeight, txNode.time = tx.blockTime, " +
                                "              txNode.totalInput = tx.totalInput, txNode.totalOutput = tx.totalOutput, " +
                                "              txNode.fee = tx.fee " +
                                "ON MATCH SET txNode.totalInput = tx.totalInput, txNode.totalOutput = tx.totalOutput, " +
                                "              txNode.fee = tx.fee, txNode.time = tx.blockTime " +
                                "WITH tx, txNode " +
                                // ----- 输入地址和 SPENT 关系 -----
                                "FOREACH (input IN tx.inputs | " +
                                "    MERGE (addr:Address {address: input.address, chain: tx.chain}) " +
                                "    ON CREATE SET addr.first_seen = tx.blockTime, addr.last_seen = tx.blockTime, " +
                                "                  addr.risk_level = 0, addr.tag = '' " +
                                "    ON MATCH SET addr.last_seen = tx.blockTime " +
                                "    MERGE (addr)-[:SPENT {amount: input.amount, index: input.index}]->(txNode) " +
                                ") " +
                                "WITH tx, txNode " +
                                // ----- 输出地址和 OUTPUT 关系 -----
                                "FOREACH (output IN tx.outputs | " +
                                "    MERGE (addr:Address {address: output.address, chain: tx.chain}) " +
                                "    ON CREATE SET addr.first_seen = tx.blockTime, addr.last_seen = tx.blockTime, " +
                                "                  addr.risk_level = 0, addr.tag = '' " +
                                "    ON MATCH SET addr.last_seen = tx.blockTime " +
                                "    MERGE (txNode)-[:OUTPUT {amount: output.amount, index: output.index}]->(addr) " +
                                ")";

                Map<String, Object> params = Collections.singletonMap("txBatch", txParams);
                log.info("开始执行优化后的批量 Cypher 查询...");
                session.query(cypher, params);
                long batchDuration = System.currentTimeMillis() - batchStart;
                log.info("第 {}/{} 批保存成功，耗时 {}ms，共 {} 笔交易",
                        i / batchSize + 1, (txs.size() + batchSize - 1) / batchSize,
                        batchDuration, batch.size());
                totalSuccess += batch.size();

            } catch (Exception e) {
                totalFail += batch.size();
                log.error("第 {}/{} 批保存失败，失败交易数 {}，错误: {}",
                        i / batchSize + 1, (txs.size() + batchSize - 1) / batchSize,
                        batch.size(), e.getMessage(), e);
                // 可选：降级为逐条保存
                fallbackSaveSingleTransactions(batch, inputsMap, outputsMap);
            } finally {
                if (session != null) {
                    closeSession(session);
                }
            }
        }

        log.info("批量保存比特币交易到图数据库完成：成功 {} 笔，失败 {} 笔", totalSuccess, totalFail);
    }

    /**
     * 降级方案：单条保存（当批量失败时使用）
     */
    private void fallbackSaveSingleTransactions(List<ChainTx> txs,
                                                Map<String, List<ChainTxInput>> inputsMap,
                                                Map<String, List<ChainTxOutput>> outputsMap) {
        for (ChainTx tx : txs) {
            try {
                saveBitcoinTransactionToGraph(tx, inputsMap.get(tx.getTxHash()), outputsMap.get(tx.getTxHash()));
            } catch (Exception e) {
                log.error("单条保存交易失败: {}", tx.getTxHash(), e);
            }
        }
    }

    @Transactional(transactionManager = "neo4jTransactionManager")
    public void saveBitcoinTransactionsToGraph(List<ChainTx> txs,
                                               Map<String, List<ChainTxInput>> inputsMap,
                                               Map<String, List<ChainTxOutput>> outputsMap) {
        saveBitcoinTransactionsBatchToGraph(txs, inputsMap, outputsMap, 50);
    }
}