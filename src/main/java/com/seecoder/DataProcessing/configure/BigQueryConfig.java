package com.seecoder.DataProcessing.configure;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.BigQueryOptions;
import com.google.cloud.http.HttpTransportOptions;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.Proxy;

@Slf4j
@Configuration
public class BigQueryConfig {

    @Value("${bigquery.project-id:dataprocessingamlsys}") // 请确保这是您的项目ID
    private String projectId;

    @Value("${bigquery.credentials-path:classpath:service-account-key.json}")
    private String credentialsPath;

    /**
     * 【核心】创建一个统一配置了代理的HttpTransportFactory
     * 这个工厂将用于构建所有Google客户端库内部的HTTP请求，包括认证和API调用。
     */
    private com.google.api.client.http.HttpTransport createProxiedHttpTransport() {
        Proxy proxy = new Proxy(Proxy.Type.HTTP, new InetSocketAddress("127.0.0.1", 7897));
        return new com.google.api.client.http.javanet.NetHttpTransport.Builder()
                .setProxy(proxy)
                .build();
    }

    @Bean
    public BigQuery bigQuery() throws IOException {
        log.info("正在配置BigQuery客户端，使用HTTP代理: 127.0.0.1:7897");

        // 1. 使用自定义的、带代理的传输层来加载凭证
        //    关键：GoogleCredentials.fromStream 内部会发起HTTP请求获取令牌，现在它也会走代理了。
        InputStream credentialsStream = new ClassPathResource("service-account-key.json").getInputStream();
        GoogleCredentials credentials = GoogleCredentials.fromStream(credentialsStream, () -> createProxiedHttpTransport());
        credentialsStream.close();

        // 2. 配置BigQuery服务本身的传输选项（查询请求走代理）
        HttpTransportOptions transportOptions = HttpTransportOptions.newBuilder()
                .setHttpTransportFactory(() -> createProxiedHttpTransport())
                .build();

        // 3. 构建客户端
        BigQuery bigQuery = BigQueryOptions.newBuilder()
                .setProjectId(projectId)
                .setCredentials(credentials)
                .setTransportOptions(transportOptions)
                .build()
                .getService();

        log.info("✅ BigQuery客户端初始化完成 (已统一启用代理)");
        return bigQuery;
    }
}