package com.example.blockchain.controller;

import com.example.blockchain.domain.*;
import com.example.blockchain.services.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

/**
 * 预测网关控制器 - 职责：接收请求，调用对应的服务
 * 
 * 依赖注入的服务（单一职责）：
 * - PredictionApiService: 预测相关 API
 * - ModelApiService: 模型信息 API
 * - HealthApiService: 健康检查和统计 API
 * - SummaryService: 摘要计算
 */
@RestController
@RequestMapping("/")
@RequiredArgsConstructor
public class ApiGatewayController {

    private final PredictionApiService predictionApiService;
    private final ModelApiService modelApiService;
    private final HealthApiService healthApiService;
    private final SummaryService summaryService;

    /**
     * 预测指定交易
     */
    @PostMapping("/predict")
    public Mono<ResponseEntity<PredictionResponse>> predictTransactions(
            @RequestBody PredictionRequest request) {
        return predictionApiService.predictTransactions(request.getTx_ids())
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(ResponseEntity.internalServerError().build()));
    }

    /**
     * 批量预测（全图）
     */
    @PostMapping("/batch_predict")
    public Mono<ResponseEntity<BatchPredictionResponse>> batchPredict() {
        return predictionApiService.batchPredict()
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(ResponseEntity.internalServerError().build()));
    }

    /**
     * 获取模型信息
     */
    @GetMapping("/model/info")
    public Mono<ResponseEntity<ModelInfo>> getModelInfo() {
        return modelApiService.getModelInfo()
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(ResponseEntity.internalServerError().build()));
    }

    /**
     * 健康检查
     */
    @GetMapping("/health")
    public Mono<ResponseEntity<HealthResponse>> healthCheck() {
        return healthApiService.healthCheck()
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(ResponseEntity.internalServerError().build()));
    }

    /**
     * 获取统计信息
     */
    @GetMapping("/statistics")
    public Mono<ResponseEntity<StatisticsResponse>> getStatistics() {
        return healthApiService.getStatistics()
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(ResponseEntity.internalServerError().build()));
    }

    /**
     * 获取预测摘要
     */
    @PostMapping("/summary")
    public Mono<ResponseEntity<PredictionSummary>> getPredictionSummary(
            @RequestBody PredictionRequest request) {
        return predictionApiService.predictTransactions(request.getTx_ids())
                .map(summaryService::calculateSummary)
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(ResponseEntity.internalServerError().build()));
    }
}
