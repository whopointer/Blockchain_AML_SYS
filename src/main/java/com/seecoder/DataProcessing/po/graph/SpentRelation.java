// 第二种关系：地址 → 交易（SPENT）
// com/seecoder/DataProcessing/po/graph/SpentRelation.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.*;
import java.math.BigDecimal;

@RelationshipEntity(type = "SPENT")
@Data
@NoArgsConstructor
public class SpentRelation {

    @Id
    @GeneratedValue
    private Long id;

    @StartNode
    private AddressNode fromAddress;

    @EndNode
    private TransactionNode transaction;

    @Property(name = "amount")
    private BigDecimal amount;

    @Property(name = "index")
    private Integer index;
}