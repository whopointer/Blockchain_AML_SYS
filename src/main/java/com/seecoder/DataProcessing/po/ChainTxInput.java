// com/seecoder/DataProcessing/po/ChainTxInput.java
package com.seecoder.DataProcessing.po;

import lombok.Data;
import javax.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "chain_tx_input")
public class ChainTxInput {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "chain", nullable = false, length = 16)
    private String chain = "BTC";

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "tx_id", nullable = false)
    private ChainTx transaction;

    @Column(name = "input_index", nullable = false)
    private Integer inputIndex;

    @Column(name = "prev_tx_hash", length = 80)
    private String prevTxHash;

    @Column(name = "prev_out_index")
    private Integer prevOutIndex;

    @Column(name = "address", length = 128)
    private String address;

    @Column(name = "value", precision = 38, scale = 8)
    private BigDecimal value;

    @Column(name = "script_sig", columnDefinition = "TEXT")
    private String scriptSig;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}