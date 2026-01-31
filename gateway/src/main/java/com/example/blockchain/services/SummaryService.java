package com.example.blockchain.services;

import com.example.blockchain.domain.PredictionResponse;
import com.example.blockchain.domain.PredictionSummary;
import com.example.blockchain.domain.RiskDistribution;
import com.example.blockchain.domain.TransactionPrediction;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 预测摘要服务 - 单一职责：计算预测结果统计摘要
 */
@Slf4j
@Service
public class SummaryService {

    /**
     * 根据预测结果计算摘要统计
     */
    public PredictionSummary calculateSummary(PredictionResponse response) {
        List<TransactionPrediction> results = response.getResults();

        if (results == null || results.isEmpty()) {
            log.warn("预测结果为空，返回空摘要");
            return new PredictionSummary();
        }

        int total = results.size();
        int suspicious = (int) results.stream()
                .filter(TransactionPrediction::is_suspicious)
                .count();
        int legitimate = total - suspicious;

        double avgConf = results.stream()
                .mapToDouble(TransactionPrediction::getConfidence)
                .average()
                .orElse(0.0);

        long highRisk = results.stream()
                .filter(r -> r.getProbability() > 0.8)
                .count();
        long mediumRisk = results.stream()
                .filter(r -> r.getProbability() >= 0.5 && r.getProbability() <= 0.8)
                .count();
        long lowRisk = total - highRisk - mediumRisk;

        RiskDistribution riskDistribution = new RiskDistribution();
        riskDistribution.setHigh_risk((int) highRisk);
        riskDistribution.setMedium_risk((int) mediumRisk);
        riskDistribution.setLow_risk((int) lowRisk);

        PredictionSummary summary = new PredictionSummary();
        summary.setTotal_transactions(total);
        summary.setSuspicious_count(suspicious);
        summary.setLegitimate_count(legitimate);
        summary.setSuspicious_rate(total > 0 ? (double) suspicious / total : 0.0);
        summary.setAverage_confidence(avgConf);
        summary.setRisk_distribution(riskDistribution);
        summary.setTimestamp(LocalDateTime.now().toString());

        log.info("摘要计算完成: total={}, suspicious={}, legitimate={}", total, suspicious, legitimate);

        return summary;
    }
}
