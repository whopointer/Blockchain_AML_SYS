// com/seecoder/DataProcessing/repository/GraphSnapshotRepository.java
package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.GraphSnapshot;
import com.seecoder.DataProcessing.enums.RiskLevel;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * 图谱快照数据库访问接口
 */
@Repository
public interface GraphSnapshotRepository extends JpaRepository<GraphSnapshot, Long> {
    // 按风险等级查询
    List<GraphSnapshot> findByRiskLevel(RiskLevel riskLevel);
}