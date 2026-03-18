package com.seecoder.DataProcessing.configure;

import com.clickhouse.jdbc.ClickHouseDataSource;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;
import java.sql.SQLException;
import java.util.Properties;

@Configuration
public class ClickHouseConfig {

    @Value("${clickhouse.jdbc-url}")
    private String url;

    @Value("${clickhouse.username}")
    private String username;

    @Value("${clickhouse.password}")
    private String password;

    @Bean
    public DataSource clickHouseDataSource() throws SQLException {
        System.out.println("ClickHouse JDBC URL: " + url); // 打印原始 URL
        Properties props = new Properties();
        props.setProperty("user", username);
        props.setProperty("password", password);
        props.setProperty("socket_timeout", "300000");
        props.setProperty("connection_timeout", "30000");
        // 确保以下参数被设置
        props.setProperty("compress", "0");
        props.setProperty("decompress", "0");
        props.setProperty("enable_http_compression", "false");
        return new ClickHouseDataSource(url, props);
    }

    @Bean
    public JdbcTemplate clickHouseJdbcTemplate(@Qualifier("clickHouseDataSource") DataSource clickHouseDataSource) {
        return new JdbcTemplate(clickHouseDataSource);
    }
}