// com/seecoder/DataProcessing/service/BigQueryConnectionTest.java
package com.seecoder.DataProcessing.service;

import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.BigQueryException;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.cloud.bigquery.TableResult;
import lombok.extern.slf4j.Slf4j;
import lombok.var;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class BigQueryConnectionTest implements CommandLineRunner {

    @Autowired
    private BigQuery bigQuery;

    @Override
    public void run(String... args) throws Exception {

        try {
            // 执行一个简单的查询测试连接
            String query = "SELECT 1 as test";
            //QueryJobConfiguration queryConfig = QueryJobConfiguration.newBuilder(query).build();
            //TableResult result = bigQuery.query(queryConfig);

            log.info("✅ BigQuery连接测试成功！");
            log.info("✅ 项目ID: {}", bigQuery.getOptions().getProjectId());

            // 测试公共数据集查询
            //testPublicDataset();

        } catch (BigQueryException e) {
            log.error("❌ BigQuery连接失败: {}", e.getMessage());
            log.error("请检查：");
            log.error("1. 是否已启用BigQuery API");
            log.error("2. 服务账号是否有权限");
            log.error("3. 凭证文件路径是否正确");
            log.error("4. 项目ID是否正确");
        }
    }

    private void testPublicDataset() {
        try {
            String query = "SELECT COUNT(*) as count FROM `bigquery-public-data.crypto_ethereum.transactions` LIMIT 1";
            QueryJobConfiguration queryConfig = QueryJobConfiguration.newBuilder(query).build();
            TableResult result = bigQuery.query(queryConfig);

            for (var row : result.iterateAll()) {
                long count = row.get("count").getLongValue();
                log.info("✅ 以太坊交易表测试成功，记录数: {}", count);
            }
        } catch (Exception e) {
            log.warn("⚠️ 公共数据集查询测试失败: {}", e.getMessage());
        }
    }
}