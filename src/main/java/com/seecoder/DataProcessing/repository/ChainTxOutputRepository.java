// com/seecoder/DataProcessing/repository/ChainTxOutputRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTxOutput;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ChainTxOutputRepository extends JpaRepository<ChainTxOutput, Long> {

    List<ChainTxOutput> findByChainAndTransaction_TxHash(String chain, String txHash);
}