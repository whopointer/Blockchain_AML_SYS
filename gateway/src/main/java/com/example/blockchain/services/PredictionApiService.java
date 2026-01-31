package com.example.blockchain.services;

import com.example.blockchain.domain.BatchPredictionResponse;
import com.example.blockchain.domain.PredictionRequest;
import com.example.blockchain.domain.PredictionResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.List;

/**
 * 预测 API 服务 - 单一职责：调用 Python 预测相关 API
 * 
 * 职责范围：
 * - 单笔交易预测
 * - 批量预测（全图）
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PredictionApiService {

    private final WebClient pythonWebClient;

    /**
     * 预测指定交易ID列表
     */
    public Mono<PredictionResponse> predictTransactions(List<String> txIds) {
        PredictionRequest request = new PredictionRequest(txIds);

        log.info("调用预测 API，交易数量: {}", txIds.size());

        return pythonWebClient.post()
                .uri("predict")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResponse.class)
                .doOnSuccess(resp -> log.info("预测成功，结果数量: {}", resp.getResults().size()))
                .doOnError(e -> log.error("预测失败: {}", e.getMessage()));
    }

    /**
     * 批量预测（全图）
     */
    public Mono<BatchPredictionResponse> batchPredict() {
        log.info("调用批量预测 API");

        return pythonWebClient.post()
                .uri("batch_predict")
                .retrieve()
                .bodyToMono(BatchPredictionResponse.class)
                .doOnSuccess(resp -> log.info("批量预测成功，统计信息: total={}", 
                    resp.getStatistics() != null ? resp.getStatistics().get("total_transactions") : "N/A"))
                .doOnError(e -> log.error("批量预测失败: {}", e.getMessage()));
    }
}
