package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.NodeEntity;
import org.neo4j.ogm.annotation.Relationship;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Data
@NodeEntity(label = "Address")
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
public class AddressNode {

    @Id
    @GeneratedValue
    @EqualsAndHashCode.Include
    private Long id;

    private String chain;

    @EqualsAndHashCode.Include
    private String address;

    private LocalDateTime firstSeen;
    private LocalDateTime lastSeen;
    private Integer riskLevel;
    private String tag;

    // 修改为List
    @Relationship(type = "TRANSFER", direction = Relationship.OUTGOING)
    private List<TransferRelation> outgoingTransfers = new ArrayList<>();

    @Relationship(type = "TRANSFER", direction = Relationship.INCOMING)
    private List<TransferRelation> incomingTransfers = new ArrayList<>();

    @Relationship(type = "SPENT", direction = Relationship.OUTGOING)
    private List<SpentRelation> spentTransactions = new ArrayList<>();

    @Relationship(type = "OUTPUT", direction = Relationship.INCOMING)
    private List<OutputRelation> receivedTransactions = new ArrayList<>();

    public AddressNode() {}

    public AddressNode(String chain, String address) {
        this.chain = chain;
        this.address = address;
    }

    // 确保集合不为null的getter方法
    public List<TransferRelation> getOutgoingTransfers() {
        if (outgoingTransfers == null) {
            outgoingTransfers = new ArrayList<>();
        }
        return outgoingTransfers;
    }

    public List<TransferRelation> getIncomingTransfers() {
        if (incomingTransfers == null) {
            incomingTransfers = new ArrayList<>();
        }
        return incomingTransfers;
    }

    public List<SpentRelation> getSpentTransactions() {
        if (spentTransactions == null) {
            spentTransactions = new ArrayList<>();
        }
        return spentTransactions;
    }

    public List<OutputRelation> getReceivedTransactions() {
        if (receivedTransactions == null) {
            receivedTransactions = new ArrayList<>();
        }
        return receivedTransactions;
    }
}