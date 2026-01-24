// com/seecoder/DataProcessing/po/ChainBlock.java
package com.seecoder.DataProcessing.po;

import lombok.Data;
import javax.persistence.*;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "chain_block")
public class ChainBlock {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "chain", length = 20, nullable = false)
    private String chain;

    @Column(name = "height", nullable = false)
    private Long height;

    @Column(name = "block_hash", length = 66, nullable = false)
    private String blockHash;

    @Column(name = "prev_block_hash", length = 66)
    private String prevBlockHash;

    @Column(name = "block_time", nullable = false)
    private LocalDateTime blockTime;

    @Column(name = "tx_count")
    private Integer txCount;

    @Column(name = "raw_size_bytes")
    private Long rawSizeBytes;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}