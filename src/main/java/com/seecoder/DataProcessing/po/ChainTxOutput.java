// com/seecoder/DataProcessing/po/ChainTxOutput.java
package com.seecoder.DataProcessing.po;

import lombok.Data;
import javax.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "chain_tx_output")
public class ChainTxOutput {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "chain", nullable = false, length = 16)
    private String chain = "BTC";

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "tx_id", nullable = false)
    private ChainTx transaction;

    @Column(name = "output_index", nullable = false)
    private Integer outputIndex;

    @Column(name = "address", length = 128)
    private String address;

    @Column(name = "value", precision = 38, scale = 8)
    private BigDecimal value;

    @Column(name = "script_pub_key", columnDefinition = "TEXT")
    private String scriptPubKey;

    @Column(name = "spent_tx_hash", length = 80)
    private String spentTxHash;

    @Column(name = "spent_time")
    private LocalDateTime spentTime;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}