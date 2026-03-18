package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainTokenTransfer;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface ChainTokenTransferRepository extends JpaRepository<ChainTokenTransfer, Long> {

    List<ChainTokenTransfer> findByTransactionHashIn(List<String> txHashes);
}