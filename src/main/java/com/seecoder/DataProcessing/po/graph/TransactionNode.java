// com/seecoder/DataProcessing/po/graph/TransactionNode.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Property;
import org.neo4j.ogm.annotation.Relationship;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

@NodeEntity(label = "Transaction")
@Data
@NoArgsConstructor
public class TransactionNode {

    @Id
    @GeneratedValue
    private Long id;

    @Property(name = "chain")
    private String chain;

    @Property(name = "tx_hash")
    private String txHash;

    @Property(name = "tx_id")
    private String txId;

    @Property(name = "block_height")
    private Long blockHeight;

    @Property(name = "time")
    private LocalDateTime time;

    @Property(name = "total_input")
    private BigDecimal totalInput;

    @Property(name = "total_output")
    private BigDecimal totalOutput;

    @Property(name = "fee")
    private BigDecimal fee;

    // 输入地址关系（更细粒度）
    @Relationship(type = "SPENT", direction = Relationship.INCOMING)
    private Set<SpentRelation> fromAddresses = new HashSet<>();

    // 输出地址关系（更细粒度）
    @Relationship(type = "OUTPUT", direction = Relationship.OUTGOING)
    private Set<OutputRelation> toAddresses = new HashSet<>();

    public TransactionNode(String chain, String txHash) {
        this.chain = chain;
        this.txHash = txHash;
    }
}