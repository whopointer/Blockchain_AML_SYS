// 为了方便，再创建一个接收关系实体（与OUTPUT相反方向）
// com/seecoder/DataProcessing/po/graph/ReceivedRelation.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.neo4j.ogm.annotation.*;
import java.math.BigDecimal;

@RelationshipEntity(type = "OUTPUT")
@Data
@NoArgsConstructor
public class ReceivedRelation {

    @Id
    @GeneratedValue
    private Long id;

    @StartNode
    private AddressNode toAddress;

    @EndNode
    private TransactionNode transaction;

    @Property(name = "amount")
    private BigDecimal amount;

    @Property(name = "index")
    private Integer index;
}