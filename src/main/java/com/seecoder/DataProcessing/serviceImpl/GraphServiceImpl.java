package com.seecoder.DataProcessing.serviceImpl;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.service.GraphService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphAddressService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphStatsService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphTransactionService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import com.seecoder.DataProcessing.po.GraphSnapshot;
import com.seecoder.DataProcessing.repository.GraphSnapshotRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
public class GraphServiceImpl implements GraphService {

    @Autowired
    private GraphTransactionService transactionService;

    @Autowired
    private GraphAddressService addressService;

    @Autowired
    private GraphStatsService statsService;

    @Autowired
    private GraphSnapshotRepository snapshotRepository;

    @Override
    public void saveTransactionToGraph(ChainTx chainTx) {
        transactionService.saveTransactionToGraph(chainTx);
    }

    @Override
    public void saveTransactionsToGraph(List<ChainTx> chainTxs) {
        transactionService.saveTransactionsToGraph(chainTxs);
    }

    @Override
    public ApiResponse<Map<String, Object>> findNhopTransactionPath(String fromAddress, String toAddress) {
        // 调用GraphAddressService的实现
        return addressService.findNhopTransactionPath(fromAddress, toAddress);
    }

    @Override
    public ApiResponse<Map<String, Object>> findAddressesWithinNHops(String address, Integer maxHops) {
        return addressService.findAddressesWithinNHops(address, maxHops);
    }

    @Override
    public ApiResponse<Map<String, Object>> getAddressTransferStats(String address) {
        return addressService.getAddressTransferStats(address);
    }

    @Override
    public ApiResponse<Map<String, Object>> getTransferStatsBetweenAddresses(String fromAddress, String toAddress) {
        return statsService.getTransferStatsBetweenAddresses(fromAddress, toAddress);
    }

    @Override
    public ApiResponse<List<Map<String, Object>>> findLargeTransfers(BigDecimal minAmount, LocalDateTime startTime, LocalDateTime endTime, Integer limit) {
        return statsService.findLargeTransfers(minAmount, startTime, endTime, limit);
    }

    @Override
    public ApiResponse<Map<String, Object>> analyzeAddressPattern(String address, Integer depth) {
        return statsService.analyzeAddressPattern(address, depth);
    }

    @Override
    public ApiResponse<Map<String, Object>> getDirectConnections(String address) {
        return addressService.getDirectConnections(address);
    }

    @Override
    public ApiResponse<Void> updateAddressRiskLevel(String address, Integer riskLevel) {
        return addressService.updateAddressRiskLevel(address, riskLevel);
    }

    @Override
    public ApiResponse<Void> tagAddress(String address, String tag) {
        return addressService.tagAddress(address, tag);
    }

    @Override
    public ApiResponse<Map<String, Object>> testNeo4jConnection() {
        return statsService.testNeo4jConnection();
    }

    @Override
    public ApiResponse<GraphSnapshot> createGraphSnapshot(GraphSnapshot snapshot) {
        // basic validation
        if (snapshot == null || snapshot.getTitle() == null) {
            return ApiResponse.error(400, "快照标题不能为空");
        }
        // 主地址优先选择中心地址，若为空则使用起始地址
        String main = snapshot.getCenterAddress();
        if ((main == null || main.isEmpty()) && snapshot.getFromAddress() != null) {
            main = snapshot.getFromAddress();
        }
        // 默认节点数/边数为0
        if (snapshot.getNodeCount() == null) {
            snapshot.setNodeCount(0);
        }
        if (snapshot.getLinkCount() == null) {
            snapshot.setLinkCount(0);
        }
        // tags字段如果有列表形式，保持原样
        GraphSnapshot saved = snapshotRepository.save(snapshot);
        return ApiResponse.success(saved, null);
    }

    @Override
    public void cleanGraphData(String chain) {
        statsService.cleanGraphData(chain);
    }

    @Override
    public ApiResponse<List<GraphSnapshot>> getAllGraphSnapshots() {
        try {
            List<GraphSnapshot> snapshots = snapshotRepository.findAll();
            return ApiResponse.success(snapshots, null);
        } catch (Exception e) {
            log.error("获取所有图谱快照失败", e);
            return ApiResponse.error(500, "获取所有图谱快照失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<GraphSnapshot> updateGraphSnapshot(Long id, GraphSnapshot snapshot) {
        try {
            // 查找快照是否存在
            GraphSnapshot existingSnapshot = snapshotRepository.findById(id).orElse(null);
            if (existingSnapshot == null) {
                return ApiResponse.error(404, "图谱快照不存在");
            }

            // 验证标题
            if (snapshot.getTitle() == null) {
                return ApiResponse.error(400, "快照标题不能为空");
            }

            // 更新字段
            existingSnapshot.setTitle(snapshot.getTitle());
            existingSnapshot.setDescription(snapshot.getDescription());
            existingSnapshot.setTags(snapshot.getTags());
            existingSnapshot.setRiskLevel(snapshot.getRiskLevel());
            existingSnapshot.setCenterAddress(snapshot.getCenterAddress());
            existingSnapshot.setFromAddress(snapshot.getFromAddress());
            existingSnapshot.setToAddress(snapshot.getToAddress());
            existingSnapshot.setHops(snapshot.getHops());
            existingSnapshot.setFilterConfig(snapshot.getFilterConfig());

            // 保存更新
            GraphSnapshot updated = snapshotRepository.save(existingSnapshot);
            return ApiResponse.success(updated, null);
        } catch (Exception e) {
            log.error("修改图谱快照信息失败", e);
            return ApiResponse.error(500, "修改图谱快照信息失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Void> deleteGraphSnapshot(Long id) {
        try {
            // 查找快照是否存在
            if (!snapshotRepository.existsById(id)) {
                return ApiResponse.error(404, "图谱快照不存在");
            }

            // 删除快照
            snapshotRepository.deleteById(id);
            return ApiResponse.success(null, null);
        } catch (Exception e) {
            log.error("删除图谱快照失败", e);
            return ApiResponse.error(500, "删除图谱快照失败: " + e.getMessage());
        }
    }
}