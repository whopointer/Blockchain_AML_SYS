package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTokenTransfer;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Set;

@Repository
public interface ChainTokenTransferRepository extends JpaRepository<ChainTokenTransfer, Long> {
    // ChainTokenTransferRepository.java
    @Query("SELECT CONCAT(tt.transactionHash, '#', tt.logIndex) FROM ChainTokenTransfer tt WHERE tt.blockTimestamp BETWEEN :start AND :end")
    Set<String> findKeysByTimeRange(@Param("start") LocalDateTime start,
                                    @Param("end") LocalDateTime end);
    List<ChainTokenTransfer> findByTransactionHashIn(List<String> txHashes);
}