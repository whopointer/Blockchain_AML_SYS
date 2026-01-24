// com/seecoder/DataProcessing/repository/graph/AddressNodeRepository.java
package com.seecoder.DataProcessing.repository.graph;

import com.seecoder.DataProcessing.po.graph.AddressNode;
import org.springframework.data.neo4j.annotation.Query;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.repository.query.Param;
import java.util.List;

public interface AddressNodeRepository extends Neo4jRepository<AddressNode, Long> {

    AddressNode findByAddress(String address);

    AddressNode findByAddressAndChain(String address, String chain);

    List<AddressNode> findByChain(String chain);

    List<AddressNode> findByRiskLevel(Integer riskLevel);

    List<AddressNode> findByTag(String tag);

    @Query("MATCH (a:Address {address: $address}) RETURN a")
    AddressNode findAddress(@Param("address") String address);

    @Query("MATCH (a:Address)-[:TRANSFER]->(b:Address) " +
            "WHERE a.address = $address " +
            "RETURN DISTINCT b.address as toAddress")
    List<String> findTransferToAddresses(@Param("address") String address);

    @Query("MATCH (a:Address)<-[:TRANSFER]-(b:Address) " +
            "WHERE a.address = $address " +
            "RETURN DISTINCT b.address as fromAddress")
    List<String> findTransferFromAddresses(@Param("address") String address);

    @Query("MATCH (a:Address {address: $address})-[r:TRANSFER]->(b:Address) " +
            "RETURN SUM(r.amount) as totalSent")
    Double getTotalSent(@Param("address") String address);

    @Query("MATCH (a:Address {address: $address})<-[r:TRANSFER]-(b:Address) " +
            "RETURN SUM(r.amount) as totalReceived")
    Double getTotalReceived(@Param("address") String address);
}