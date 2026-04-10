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

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.Proxy;

@Slf4j
@Configuration
public class BigQueryConfig {

    @Value("${bigquery.project-id:dataprocessingamlsys}")
    private String projectId;

    @Value("${bigquery.credentials-path:classpath:service-account-key.json}")
    private String credentialsPath;

    /**
     * 【核心】创建一个统一配置了代理的HttpTransportFactory
     * 这个工厂将用于构建所有Google客户端库内部的HTTP请求，包括认证和API调用。
     */
    private com.google.api.client.http.HttpTransport createProxiedHttpTransport() {
        Proxy proxy = new Proxy(Proxy.Type.HTTP, new InetSocketAddress("127.0.0.1", 7890));
        return new com.google.api.client.http.javanet.NetHttpTransport.Builder()
                .setProxy(proxy)
                .build();
    }

    @Bean
    public BigQuery bigQuery() {
        try {
            log.info("正在配置BigQuery客户端，使用HTTP代理: 127.0.0.1:7890");

            // 1. 根据路径类型获取凭证输入流
            InputStream credentialsStream = getCredentialsInputStream(credentialsPath);
            if (credentialsStream == null) {
                throw new FileNotFoundException("无法找到凭证文件: " + credentialsPath);
            }

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
        } catch (Exception e) {
            log.warn("BigQuery客户端配置失败: {}", e.getMessage());
            log.warn("在开发环境中可以暂时跳过BigQuery配置");
            // 在开发环境中返回null，应用程序将使用模拟数据
            return null;
        }
    }

    /**
     * 根据路径字符串获取输入流，支持以下格式：
     * - classpath:xxx  -> 从类路径加载
     * - /xxx 或 C:/xxx -> 绝对路径
     * - xxx             -> 相对路径（尝试类路径，若失败则尝试文件系统相对路径）
     */
    private InputStream getCredentialsInputStream(String path) throws IOException {
        if (path == null || path.isEmpty()) {
            throw new FileNotFoundException("凭证路径为空");
        }

        // 1. 明确 classpath: 前缀
        if (path.startsWith("classpath:")) {
            String classpathLocation = path.substring("classpath:".length());
            ClassPathResource resource = new ClassPathResource(classpathLocation);
            if (resource.exists()) {
                return resource.getInputStream();
            } else {
                throw new FileNotFoundException("Classpath resource不存在: " + classpathLocation);
            }
        }

        // 2. 绝对路径（Unix 或 Windows 盘符）
        if (path.startsWith("/") || path.matches("^[A-Za-z]:.*")) {
            return new FileInputStream(path);
        }

        // 3. 既不是classpath也不是绝对路径，先尝试作为类路径资源，再尝试作为相对文件路径
        ClassPathResource classPathResource = new ClassPathResource(path);
        if (classPathResource.exists()) {
            return classPathResource.getInputStream();
        }

        // 最后尝试相对文件路径
        return new FileInputStream(path);
    }
}