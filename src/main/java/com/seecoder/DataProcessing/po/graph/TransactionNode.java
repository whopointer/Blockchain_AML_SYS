// 修改 TransactionNode.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Relationship;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

@Data
@NodeEntity(label = "Transaction")
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
public class TransactionNode {

    @Id
    @GeneratedValue
    @EqualsAndHashCode.Include
    private Long id;

    private String chain;

    @EqualsAndHashCode.Include
    private String txHash;

    private Long blockHeight;
    private LocalDateTime time;
    private Double totalInput;     // 修改：BigDecimal -> Double
    private Double totalOutput;    // 修改：BigDecimal -> Double
    private Double fee;            // 修改：BigDecimal -> Double

    @Relationship(type = "SPENT", direction = Relationship.INCOMING)
    private Set<SpentRelation> fromAddresses = new HashSet<>();

    @Relationship(type = "OUTPUT", direction = Relationship.OUTGOING)
    private Set<OutputRelation> toAddresses = new HashSet<>();

    public TransactionNode() {}

    public TransactionNode(String chain, String txHash) {
        this.chain = chain;
        this.txHash = txHash;
    }
}