// com/seecoder/DataProcessing/configure/Neo4jConfig.java
package com.seecoder.DataProcessing.configure;

import org.neo4j.ogm.config.Configuration;
import org.neo4j.ogm.session.SessionFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.data.neo4j.repository.config.EnableNeo4jRepositories;
import org.springframework.data.neo4j.transaction.Neo4jTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@org.springframework.context.annotation.Configuration
@EnableNeo4jRepositories(
        basePackages = "com.seecoder.DataProcessing.repository.graph",
        sessionFactoryRef = "neo4jSessionFactory"
)
@EnableTransactionManagement
public class Neo4jConfig {

    // 修改：使用 application.yml 中实际的配置属性名
    @Value("${spring.neo4j.uri:bolt://localhost:7687}")
    private String neo4jUri;

    @Value("${spring.neo4j.authentication.username:neo4j}")
    private String username;

    @Value("${spring.neo4j.authentication.password:123456}")
    private String password;

    @Bean
    public Configuration configuration() {
        Configuration configuration = new Configuration.Builder()
                .uri(neo4jUri)
                .credentials(username, password)
                .autoIndex("update")
                .build();
        return configuration;
    }

    @Bean
    public SessionFactory neo4jSessionFactory() {
        return new SessionFactory(configuration(),
                "com.seecoder.DataProcessing.po.graph");
    }

    @Bean
    @Primary
    public Neo4jTransactionManager neo4jTransactionManager() {
        return new Neo4jTransactionManager(neo4jSessionFactory());
    }
}