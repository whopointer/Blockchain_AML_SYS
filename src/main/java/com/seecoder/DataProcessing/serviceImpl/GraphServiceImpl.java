package com.seecoder.DataProcessing.serviceImpl;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.ChainTxInput;
import com.seecoder.DataProcessing.po.ChainTxOutput;
import com.seecoder.DataProcessing.service.GraphService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphAddressService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphStatsService;
import com.seecoder.DataProcessing.serviceImpl.graph.GraphTransactionService;
import com.seecoder.DataProcessing.vo.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
@Service
public class GraphServiceImpl implements GraphService {

    @Autowired
    private GraphTransactionService transactionService;

    @Autowired
    private GraphAddressService addressService;

    @Autowired
    private GraphStatsService statsService;

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
    public void cleanGraphData(String chain) {
        statsService.cleanGraphData(chain);
    }

    // ============ 补充实现 GraphService 接口缺失的方法 ============

    @Autowired
    private com.seecoder.DataProcessing.repository.GraphSnapshotRepository graphSnapshotRepository; // 请确保该 Repository 存在

    @Override
    public ApiResponse<com.seecoder.DataProcessing.po.GraphSnapshot> createGraphSnapshot(com.seecoder.DataProcessing.po.GraphSnapshot snapshot) {
        try {
            com.seecoder.DataProcessing.po.GraphSnapshot saved = graphSnapshotRepository.save(snapshot);
            return ApiResponse.success(saved, null);
        } catch (Exception e) {
            log.error("创建图谱快照失败", e);
            return ApiResponse.error(500, "创建快照失败: " + e.getMessage());
        }
    }

    @Override
    public Set<String> getNeighborAddresses(String address, LocalDateTime startTime, LocalDateTime endTime) {
        return addressService.getNeighborAddresses(address, startTime, endTime);
    }

    @Override
    public Set<String> getTransactionHashes(String address, LocalDateTime startTime, LocalDateTime endTime) {
        return addressService.getTransactionHashes(address, startTime, endTime);
    }

    @Override
    public void saveBitcoinTransactionToGraph(ChainTx tx, List<ChainTxInput> inputs, List<ChainTxOutput> outputs) {
        transactionService.saveBitcoinTransactionToGraph(tx, inputs, outputs);
    }

    @Override
    public void saveBitcoinTransactionsToGraph(List<ChainTx> txs,
                                               Map<String, List<ChainTxInput>> inputsMap,
                                               Map<String, List<ChainTxOutput>> outputsMap) {
        transactionService.saveBitcoinTransactionsToGraph(txs, inputsMap, outputsMap);
    }

    @Override
    public ApiResponse<List<com.seecoder.DataProcessing.po.GraphSnapshot>> getAllGraphSnapshots() {
        try {
            List<com.seecoder.DataProcessing.po.GraphSnapshot> snapshots = graphSnapshotRepository.findAll();
            return ApiResponse.success(snapshots, (long) snapshots.size());
        } catch (Exception e) {
            log.error("获取所有图谱快照失败", e);
            return ApiResponse.error(500, "获取快照列表失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<com.seecoder.DataProcessing.po.GraphSnapshot> updateGraphSnapshot(Long id, com.seecoder.DataProcessing.po.GraphSnapshot snapshot) {
        try {
            if (!graphSnapshotRepository.existsById(id)) {
                return ApiResponse.error(404, "快照不存在");
            }
            snapshot.setId(id); // 确保 ID 正确
            com.seecoder.DataProcessing.po.GraphSnapshot updated = graphSnapshotRepository.save(snapshot);
            return ApiResponse.success(updated, null);
        } catch (Exception e) {
            log.error("更新图谱快照失败", e);
            return ApiResponse.error(500, "更新快照失败: " + e.getMessage());
        }
    }

    @Override
    public ApiResponse<Void> deleteGraphSnapshot(Long id) {
        try {
            if (!graphSnapshotRepository.existsById(id)) {
                return ApiResponse.error(404, "快照不存在");
            }
            graphSnapshotRepository.deleteById(id);
            return ApiResponse.success(null, null);
        } catch (Exception e) {
            log.error("删除图谱快照失败", e);
            return ApiResponse.error(500, "删除快照失败: " + e.getMessage());
        }
    }

    @Override
    public void saveTransactionsBatchToGraph(List<ChainTx> txs){
        transactionService.saveTransactionsBatchToGraph(txs);
    }

    @Override
    public ApiResponse<Map<String, Object>> findBTCAddressesWithinNHops(String address, Integer maxHops) {
        return addressService.findBTCAddressesWithinNHops(address, maxHops);
    }

    @Override
    public ApiResponse<Map<String, Object>> findBTCNhopTransactionPath(String fromAddress, String toAddress) {
        return addressService.findBTCNhopTransactionPath(fromAddress, toAddress);
    }

}