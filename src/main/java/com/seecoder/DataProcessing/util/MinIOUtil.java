// com/seecoder/DataProcessing/util/MinIOUtil.java
package com.seecoder.DataProcessing.util;

import com.seecoder.DataProcessing.vo.ApiResponse;
import io.minio.*;
import io.minio.errors.*;
import io.minio.http.Method;
import io.minio.messages.Item;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Slf4j
@Component
public class MinIOUtil {

    @Autowired
    private MinioClient minioClient;

    @Value("${minio.bucket-name}")
    private String bucketName;

    private static final DateTimeFormatter DATE_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy/MM/dd");
    private static final DateTimeFormatter TIMESTAMP_FORMATTER =
            DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS");


    @PostConstruct
    public void init() {

        log.info("MinIO工具初始化完成, bucket: {}", bucketName);
    }
    /**
     * 检查并创建桶（如果不存在）
     */
    private void ensureBucketExists() throws Exception {

        boolean found = minioClient.bucketExists(BucketExistsArgs.builder()
                .bucket(bucketName)
                .build());

        if (!found) {
            minioClient.makeBucket(MakeBucketArgs.builder()
                    .bucket(bucketName)
                    .build());
            log.info("创建MinIO桶: {}", bucketName);
        }
    }

    /**
     * 上传JSON数据到MinIO
     */
    public ApiResponse<String> uploadJsonData(String dataType, String fileName, String jsonContent) {
        if (minioClient == null) {
            log.warn("MinIO客户端未初始化，跳过上传");
            return ApiResponse.success("MinIO未启用", null);
        }

        try {
            ensureBucketExists();

            // 生成存储路径：data_type/yyyy/MM/dd/fileName.json
            String datePath = LocalDate.now().format(DATE_FORMATTER);
            String objectName = String.format("%s/%s/%s.json",
                    dataType, datePath, fileName);

            byte[] content = jsonContent.getBytes(StandardCharsets.UTF_8);
            InputStream inputStream = new ByteArrayInputStream(content);

            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .stream(inputStream, content.length, -1)
                    .contentType("application/json")
                    .build());

            log.info("JSON数据已上传到MinIO: {}", objectName);
            return ApiResponse.success(objectName, null);

        } catch (Exception e) {
            log.error("上传JSON数据到MinIO失败", e);
            return ApiResponse.error(500, "上传JSON数据失败: " + e.getMessage());
        }
    }

    /**
     * 上传日志文件到MinIO
     */
    public ApiResponse<String> uploadLogFile(String logType, String logContent) {
        if (minioClient == null) {
            log.warn("MinIO客户端未初始化，跳过上传");
            return ApiResponse.success("MinIO未启用", null);
        }

        try {
            ensureBucketExists();

            // 生成日志文件名：logs/log_type/yyyy/MM/dd/yyyyMMdd_HHmmss_SSS.log
            String datePath = LocalDate.now().format(DATE_FORMATTER);
            String timestamp = java.time.LocalDateTime.now().format(TIMESTAMP_FORMATTER);
            String objectName = String.format("logs/%s/%s/%s.log",
                    logType, datePath, timestamp);

            byte[] content = logContent.getBytes(StandardCharsets.UTF_8);
            InputStream inputStream = new ByteArrayInputStream(content);

            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .stream(inputStream, content.length, -1)
                    .contentType("text/plain")
                    .build());

            log.info("日志文件已上传到MinIO: {}", objectName);
            return ApiResponse.success(objectName, null);

        } catch (Exception e) {
            log.error("上传日志文件到MinIO失败", e);
            return ApiResponse.error(500, "上传日志文件失败: " + e.getMessage());
        }
    }

    /**
     * 上传原始区块链数据JSON
     */
    public ApiResponse<String> uploadBlockchainData(String chain, String dataType,
                                                    Long blockHeight, String jsonContent) {
        if (minioClient == null) {
            log.warn("MinIO客户端未初始化，跳过上传");
            return ApiResponse.success("MinIO未启用", null);
        }

        try {
            ensureBucketExists();

            // 生成存储路径：blockchain/{chain}/{data_type}/{height_group}/block_{height}.json
            long heightGroup = (blockHeight / 10000) * 10000; // 每10000个块一组
            String objectName = String.format("blockchain/%s/%s/%d/block_%d.json",
                    chain, dataType, heightGroup, blockHeight);

            byte[] content = jsonContent.getBytes(StandardCharsets.UTF_8);
            InputStream inputStream = new ByteArrayInputStream(content);

            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .stream(inputStream, content.length, -1)
                    .contentType("application/json")
                    .build());

            log.debug("区块链数据已上传到MinIO: {}", objectName);
            return ApiResponse.success(objectName, null);

        } catch (Exception e) {
            log.error("上传区块链数据到MinIO失败", e);
            return ApiResponse.error(500, "上传区块链数据失败: " + e.getMessage());
        }
    }

    /**
     * 从MinIO下载文件
     */
    public ApiResponse<byte[]> downloadFile(String objectName) {


        try (GetObjectResponse response = minioClient.getObject(GetObjectArgs.builder()
                .bucket(bucketName)
                .object(objectName)
                .build())) {

            // 使用 java.io.ByteArrayOutputStream
            java.io.ByteArrayOutputStream outputStream = new java.io.ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = response.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            byte[] content = outputStream.toByteArray();

            return ApiResponse.success(content, (long) content.length);

        } catch (Exception e) {
            log.error("从MinIO下载文件失败: {}", objectName, e);
            return ApiResponse.error(500, "下载文件失败: " + e.getMessage());
        }
    }

    /**
     * 获取文件的预签名URL（用于临时访问）
     */
    public ApiResponse<String> getPresignedUrl(String objectName, int expiryHours) {
        try {
            String url = minioClient.getPresignedObjectUrl(
                    GetPresignedObjectUrlArgs.builder()
                            .method(Method.GET)
                            .bucket(bucketName)
                            .object(objectName)
                            .expiry(expiryHours * 60 * 60) // 转换为秒
                            .build());

            return ApiResponse.success(url, null);

        } catch (Exception e) {
            log.error("生成预签名URL失败: {}", objectName, e);
            return ApiResponse.error(500, "生成预签名URL失败: " + e.getMessage());
        }
    }

    /**
     * 删除文件
     */
    public ApiResponse<String> deleteFile(String objectName) {
        try {
            minioClient.removeObject(RemoveObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build());

            log.info("已从MinIO删除文件: {}", objectName);
            return ApiResponse.success("文件删除成功", null);

        } catch (Exception e) {
            log.error("从MinIO删除文件失败: {}", objectName, e);
            return ApiResponse.error(500, "删除文件失败: " + e.getMessage());
        }
    }

    /**
     * 列出指定前缀的文件
     */
    public ApiResponse<Map<String, Object>> listFiles(String prefix, boolean recursive) {
        try {
            Iterable<Result<io.minio.messages.Item>> results = minioClient.listObjects(
                    ListObjectsArgs.builder()
                            .bucket(bucketName)
                            .prefix(prefix)
                            .recursive(recursive)
                            .build());

            Map<String, Object> fileList = new LinkedHashMap<>();
            int count = 0;

            for (Result<io.minio.messages.Item> result : results) {
                try {
                    io.minio.messages.Item item = result.get();
                    Map<String, Object> fileInfo = new HashMap<>();
                    fileInfo.put("objectName", item.objectName());
                    fileInfo.put("size", item.size());
                    fileInfo.put("lastModified", item.lastModified().toLocalDateTime().toString());
                    fileList.put(String.valueOf(count++), fileInfo);
                } catch (Exception e) {
                    log.warn("处理文件项失败: {}", e.getMessage());
                }
            }

            Map<String, Object> result = new HashMap<>();
            result.put("totalCount", count);
            result.put("files", fileList);

            return ApiResponse.success(result, (long) count);

        } catch (Exception e) {
            log.error("列出MinIO文件失败", e);
            return ApiResponse.error(500, "列出文件失败: " + e.getMessage());
        }
    }

    /**
     * 检查文件是否存在
     */
    public boolean fileExists(String objectName) {
        try {
            minioClient.statObject(StatObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build());
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 获取文件信息
     */
    public ApiResponse<Map<String, Object>> getFileInfo(String objectName) {
        try {
            StatObjectResponse stat = minioClient.statObject(StatObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build());

            Map<String, Object> fileInfo = new HashMap<>();
            fileInfo.put("objectName", objectName);
            fileInfo.put("size", stat.size());
            fileInfo.put("lastModified", stat.lastModified().toLocalDateTime().toString());
            fileInfo.put("etag", stat.etag());
            fileInfo.put("contentType", stat.contentType());

            return ApiResponse.success(fileInfo, null);

        } catch (Exception e) {
            log.error("获取文件信息失败: {}", objectName, e);
            return ApiResponse.error(500, "获取文件信息失败: " + e.getMessage());
        }
    }

    // 添加ByteArrayOutputStream的导入
    static class ByteArrayOutputStream extends java.io.ByteArrayOutputStream {
        // 使用默认实现
    }
}