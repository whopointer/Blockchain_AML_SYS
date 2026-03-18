package com.seecoder.DataProcessing;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication  // 不再排除任何自动配置
@EnableScheduling
@EnableAsync
@EnableCaching
public class DataProcessingApplication {
	public static void main(String[] args) {
		SpringApplication.run(DataProcessingApplication.class, args);
	}
}