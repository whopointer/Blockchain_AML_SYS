// 第一种关系：地址 → 地址
// com/seecoder/DataProcessing/po/graph/TransferRelation.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@RelationshipEntity(type = "TRANSFER")
@Data
@NoArgsConstructor
public class TransferRelation {

    @Id
    @GeneratedValue
    private Long id;

    @StartNode
    private AddressNode fromAddress;

    @EndNode
    private AddressNode toAddress;

    @Property(name = "tx_hash")
    private String txHash;

    @Property(name = "amount")
    private BigDecimal amount;

    @Property(name = "time")
    private LocalDateTime time;
}