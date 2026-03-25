package com.seecoder.DataProcessing.po;

import lombok.Data;
import javax.persistence.*;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "blacklist_address")
public class BlacklistAddress {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private String address;

    private LocalDateTime createdAt = LocalDateTime.now();
}