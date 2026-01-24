// com/seecoder/DataProcessing/repository/ChainBlockRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.ChainBlock;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface ChainBlockRepository extends JpaRepository<ChainBlock, Long> {

    // 基本查询方法
    Optional<ChainBlock> findByChainAndHeight(String chain, Long height);

    // 分页查询
    Page<ChainBlock> findByChain(String chain, Pageable pageable);

    // 列表查询（用于非分页场景）
    List<ChainBlock> findAllByChain(String chain, Pageable pageable);

    Long countByChain(String chain);

    // 高度范围查询
    List<ChainBlock> findByChainAndHeightBetween(String chain, Long startHeight, Long endHeight, Pageable pageable);

    List<ChainBlock> findByChainAndHeightGreaterThanEqual(String chain, Long startHeight, Pageable pageable);

    List<ChainBlock> findByChainAndHeightLessThanEqual(String chain, Long endHeight, Pageable pageable);

    // 时间范围查询
    List<ChainBlock> findByChainAndBlockTimeBetween(String chain, LocalDateTime startTime, LocalDateTime endTime, Pageable pageable);

    Long countByChainAndBlockTimeBetween(String chain, LocalDateTime startTime, LocalDateTime endTime);

    // 聚合查询
    @Query("SELECT MAX(b.height) FROM ChainBlock b WHERE b.chain = :chain")
    Long findMaxHeight(@Param("chain") String chain);

    @Query("SELECT MAX(b.blockTime) FROM ChainBlock b WHERE b.chain = :chain")
    LocalDateTime findLatestBlockTime(@Param("chain") String chain);

    // 批量查询
    List<ChainBlock> findByChainAndHeightIn(String chain, List<Long> heights);
}