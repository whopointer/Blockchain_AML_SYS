package com.example.blockchain.services;

import com.example.blockchain.domain.ModelInfo;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

/**
 * 模型 API 服务 - 单一职责：调用 Python 模型相关 API
 * 
 * 职责范围：
 * - 获取模型信息
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ModelApiService {

    private final WebClient pythonWebClient;

    /**
     * 获取模型信息
     */
    public Mono<ModelInfo> getModelInfo() {
        log.info("调用获取模型信息 API");

        return pythonWebClient.get()
                .uri("model/info")
                .retrieve()
                .bodyToMono(ModelInfo.class)
                .doOnSuccess(info -> log.info("获取模型信息成功: type={}", info.getModel_type()))
                .doOnError(e -> log.error("获取模型信息失败: {}", e.getMessage()));
    }
}
