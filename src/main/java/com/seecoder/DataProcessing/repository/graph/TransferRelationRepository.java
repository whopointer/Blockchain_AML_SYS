// com/seecoder/DataProcessing/repository/graph/TransferRelationRepository.java
package com.seecoder.DataProcessing.repository.graph;

import com.seecoder.DataProcessing.po.graph.TransferRelation;
import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.repository.query.Param;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

public interface TransferRelationRepository extends Neo4jRepository<TransferRelation, Long> {

    TransferRelation findByTxHash(String txHash);

    List<TransferRelation> findByFromAddress_Address(String fromAddress);

    List<TransferRelation> findByToAddress_Address(String toAddress);

    List<TransferRelation> findByAmountGreaterThan(BigDecimal amount);

    @Query("MATCH (a:Address {address: $fromAddress})-[r:TRANSFER]->(b:Address {address: $toAddress}) RETURN r")
    List<TransferRelation> findTransfersBetween(@Param("fromAddress") String fromAddress,
                                                @Param("toAddress") String toAddress);

    @Query("MATCH path = (a:Address {address: $address})-[r:TRANSFER*1..$hops]->(b:Address) " +
            "RETURN b.address as address, " +
            "       LENGTH(path) as distance, " +
            "       REDUCE(total = 0.0, rel IN r | total + rel.amount) as totalAmount")
    List<Map<String, Object>> findAddressesWithinHops(@Param("address") String address,
                                                      @Param("hops") Integer hops);

    @Query("MATCH path = shortestPath((a:Address {address: $fromAddress})-[r:TRANSFER*1..$maxHops]->(b:Address {address: $toAddress})) " +
            "RETURN NODES(path) as nodes, RELATIONSHIPS(path) as rels, LENGTH(path) as hopCount")
    List<Map<String, Object>> findShortestPath(@Param("fromAddress") String fromAddress,
                                               @Param("toAddress") String toAddress,
                                               @Param("maxHops") Integer maxHops);

    @Query("MATCH (a:Address)-[r:TRANSFER]->(b:Address) " +
            "WHERE r.time >= $startTime AND r.time <= $endTime " +
            "RETURN a.address as fromAddress, b.address as toAddress, r.amount as amount, r.tx_hash as txHash, r.time as time " +
            "ORDER BY r.amount DESC " +
            "LIMIT $limit")
    List<Map<String, Object>> findLargeTransfers(@Param("startTime") String startTime,
                                                 @Param("endTime") String endTime,
                                                 @Param("limit") Integer limit);
}