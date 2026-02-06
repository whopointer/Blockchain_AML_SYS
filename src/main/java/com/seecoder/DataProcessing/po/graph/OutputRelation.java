// 修改 OutputRelation.java
package com.seecoder.DataProcessing.po.graph;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.neo4j.ogm.annotation.EndNode;
import org.neo4j.ogm.annotation.GeneratedValue;
import org.neo4j.ogm.annotation.Id;
import org.neo4j.ogm.annotation.RelationshipEntity;
import org.neo4j.ogm.annotation.StartNode;

@Data
@RelationshipEntity(type = "OUTPUT")
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
public class OutputRelation {

    @Id
    @GeneratedValue
    @EqualsAndHashCode.Include
    private Long id;

    @StartNode
    private TransactionNode transaction;

    @EndNode
    private AddressNode toAddress;

    private Double amount;  // 修改：BigDecimal -> Double
    private Integer index;
}