package com.seecoder.DataProcessing.configure;

import io.minio.MinioClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MinIOConfig {

    @Value("${minio.endpoint}")
    private String endpoint;

    @Value("${minio.access-key}")  // 注意：这里要和配置文件中一致（带连字符）
    private String accessKey;

    @Value("${minio.secret-key}")  // 注意：这里要和配置文件中一致（带连字符）
    private String secretKey;

    @Value("${minio.bucket-name}")  // 注意：这里要和配置文件中一致（带连字符）
    private String bucketName;

    @Bean
    public MinioClient minioClient() {
        try {
            log.info("正在初始化MinIO客户端...");
            log.info("MinIO配置: endpoint={}, bucket={}, accessKey={}", endpoint, bucketName, accessKey);

            MinioClient client = MinioClient.builder()
                    .endpoint(endpoint)
                    .credentials(accessKey, secretKey)
                    .build();

            log.info("✅ MinIO客户端初始化成功");
            return client;

        } catch (Exception e) {
            log.error("❌ 初始化MinIO客户端失败", e);
            throw new RuntimeException("初始化MinIO客户端失败: " + e.getMessage(), e);
        }
    }

    // 添加getter方法供其他组件使用
    public String getBucketName() {
        return bucketName;
    }
}