package com.seecoder.DataProcessing.serviceImpl;

import com.seecoder.DataProcessing.po.ChainTx;
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
    public ApiResponse<Map<String, Object>> findNhopTransactionPath(String fromAddress, String toAddress, Integer maxHops) {
        // 调用GraphAddressService的实现
        return addressService.findNhopTransactionPath(fromAddress, toAddress, maxHops);
    }

    @Override
    public ApiResponse<List<Map<String, Object>>> findAddressesWithinNHops(String address, Integer maxHops) {
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
}