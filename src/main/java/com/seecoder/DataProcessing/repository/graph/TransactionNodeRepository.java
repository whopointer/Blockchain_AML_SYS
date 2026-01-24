// com/seecoder/DataProcessing/repository/graph/TransactionNodeRepository.java
package com.seecoder.DataProcessing.repository.graph;

import com.seecoder.DataProcessing.po.graph.TransactionNode;
import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.repository.query.Param;
import java.util.List;

public interface TransactionNodeRepository extends Neo4jRepository<TransactionNode, Long> {

    TransactionNode findByTxHash(String txHash);

    TransactionNode findByTxHashAndChain(String txHash, String chain);

    List<TransactionNode> findByBlockHeight(Long blockHeight);

    List<TransactionNode> findByChain(String chain);

    @Query("MATCH (t:Transaction) WHERE t.time >= $startTime AND t.time <= $endTime RETURN t")
    List<TransactionNode> findByTimeRange(@Param("startTime") String startTime,
                                          @Param("endTime") String endTime);

    @Query("MATCH (t:Transaction) WHERE t.block_height >= $startHeight AND t.block_height <= $endHeight RETURN t")
    List<TransactionNode> findByBlockHeightRange(@Param("startHeight") Long startHeight,
                                                 @Param("endHeight") Long endHeight);
}