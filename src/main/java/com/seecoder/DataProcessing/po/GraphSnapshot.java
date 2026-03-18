// com/seecoder/DataProcessing/po/GraphSnapshot.java
package com.seecoder.DataProcessing.po;

import com.seecoder.DataProcessing.enums.RiskLevel;
import lombok.Data;

import javax.persistence.*;
import java.time.LocalDateTime;

/**
 * 图谱快照实体，映射到 mysql 的 graph_snapshot 表。
 * 通过 JPA 的 ddl-auto=update 模式会自动创建或更新对应表结构。
 */
@Data
@Entity
@Table(name = "graph_snapshot")
public class GraphSnapshot {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "title", length = 200, nullable = false)
    private String title;

    @Column(name = "description", length = 1000)
    private String description;

    /**
     * 标签数组，前端直接传入字符串数组。
     * 在数据库中使用单独的关联表存储。
     */
    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(name = "graph_snapshot_tags", joinColumns = @JoinColumn(name = "snapshot_id"))
    @Column(name = "tag", length = 100)
    private java.util.List<String> tags;

    @Column(name = "create_time")
    private LocalDateTime createTime;

    @Column(name = "node_count")
    private Integer nodeCount;

    @Column(name = "link_count")
    private Integer linkCount;

    @Column(name = "risk_level", length = 20)
    @Enumerated(EnumType.STRING)
    private RiskLevel riskLevel;

    @Column(name = "center_address", length = 100)
    private String centerAddress;

    @Column(name = "from_address", length = 100)
    private String fromAddress;

    @Column(name = "to_address", length = 100)
    private String toAddress;

    @Column(name = "hops")
    private Integer hops;

    /**
     * 嵌入的过滤配置，包含 txType、addrType 等条件。
     */
    @Embedded
    private FilterConfig filterConfig;

    @PrePersist
    protected void onCreate() {
        if (createTime == null) {
            createTime = LocalDateTime.now();
        }
    }
}