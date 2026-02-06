// 修改 TransferRelation.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.neo4j.ogm.annotation.EndNode;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.RelationshipEntity;
import org.neo4j.ogm.annotation.StartNode;

import java.time.LocalDateTime;

@Data
@RelationshipEntity(type = "TRANSFER")
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
public class TransferRelation {

    @Id
    @GeneratedValue
    @EqualsAndHashCode.Include
    private Long id;

    @StartNode
    private AddressNode fromAddress;

    @EndNode
    private AddressNode toAddress;

    private String txHash;
    private Double amount;  // 修改：BigDecimal -> Double
    private LocalDateTime time;
}