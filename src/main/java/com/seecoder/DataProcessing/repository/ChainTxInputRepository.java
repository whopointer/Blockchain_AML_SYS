// com/seecoder/DataProcessing/repository/ChainTxInputRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTxInput;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ChainTxInputRepository extends JpaRepository<ChainTxInput, Long> {

    List<ChainTxInput> findByChainAndTransaction_TxHash(String chain, String txHash);
}