// 第二种关系：交易 → 地址（OUTPUT）
// com/seecoder/DataProcessing/po/graph/OutputRelation.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.*;
import java.math.BigDecimal;

@RelationshipEntity(type = "OUTPUT")
@Data
@NoArgsConstructor
public class OutputRelation {

    @Id
    @GeneratedValue
    private Long id;

    @StartNode
    private TransactionNode transaction;

    @EndNode
    private AddressNode toAddress;

    @Property(name = "amount")
    private BigDecimal amount;

    @Property(name = "index")
    private Integer index;
}