package com.example.blockchain.services;

import com.example.blockchain.domain.HealthResponse;
import com.example.blockchain.domain.StatisticsResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

/**
 * 健康检查 API 服务 - 单一职责：调用 Python 健康检查和统计 API
 * 
 * 职责范围：
 * - 健康检查
 * - 获取统计信息
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class HealthApiService {

    private final WebClient pythonWebClient;

    /**
     * 健康检查
     */
    public Mono<HealthResponse> healthCheck() {
        log.info("调用健康检查 API");

        return pythonWebClient.get()
                .uri("health")
                .retrieve()
                .bodyToMono(HealthResponse.class)
                .doOnSuccess(health -> log.info("健康检查完成: status={}", health.getStatus()))
                .doOnError(e -> log.error("健康检查失败: {}", e.getMessage()));
    }

    /**
     * 获取统计信息
     */
    public Mono<StatisticsResponse> getStatistics() {
        log.info("调用获取统计信息 API");

        return pythonWebClient.get()
                .uri("statistics")
                .retrieve()
                .bodyToMono(StatisticsResponse.class)
                .doOnSuccess(stats -> log.info("获取统计信息成功: status={}", stats.getSystem_status()))
                .doOnError(e -> log.error("获取统计信息失败: {}", e.getMessage()));
    }
}
