// com/seecoder/DataProcessing/po/graph/AddressNode.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Property;
import org.neo4j.ogm.annotation.Relationship;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

@NodeEntity(label = "Address")
@Data
@NoArgsConstructor
public class AddressNode {

    @Id
    @GeneratedValue
    private Long id;

    @Property(name = "chain")
    private String chain;

    @Property(name = "address")
    private String address;

    @Property(name = "address_id")
    private String addressId;

    @Property(name = "first_seen")
    private LocalDateTime firstSeen;

    @Property(name = "last_seen")
    private LocalDateTime lastSeen;

    @Property(name = "tag")
    private String tag;

    @Property(name = "risk_level")
    private Integer riskLevel;

    // 出边关系：地址发起的转账
    @Relationship(type = "TRANSFER", direction = Relationship.OUTGOING)
    private Set<TransferRelation> outgoingTransfers = new HashSet<>();

    // 入边关系：地址接收的转账
    @Relationship(type = "TRANSFER", direction = Relationship.INCOMING)
    private Set<TransferRelation> incomingTransfers = new HashSet<>();

    // 发送的交易关系（更细粒度）
    @Relationship(type = "SPENT", direction = Relationship.OUTGOING)
    private Set<SpentRelation> spentTransactions = new HashSet<>();

    // 接收的交易关系（更细粒度）- OUTPUT关系的反向
    @Relationship(type = "OUTPUT", direction = Relationship.INCOMING)
    private Set<OutputRelation> receivedTransactions = new HashSet<>();

    public AddressNode(String chain, String address) {
        this.chain = chain;
        this.address = address;
        this.firstSeen = LocalDateTime.now();
        this.lastSeen = LocalDateTime.now();
        this.riskLevel = 0;
    }
}