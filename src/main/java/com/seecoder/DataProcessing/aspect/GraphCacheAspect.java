package com.seecoder.DataProcessing.aspect;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;

@Slf4j
@Aspect
@Component
public class GraphCacheAspect {

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    private static final String CACHE_PREFIX = "graph:cache:";
    private static final long CACHE_TTL_MINUTES = 30;

    @Pointcut("execution(* com.seecoder.DataProcessing.controller.Neo4jController.*(..))")
    public void graphControllerPointcut() {
    }

    @Around("graphControllerPointcut()")
    public Object cacheGraphQuery(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().getName();

        if (!isCacheableMethod(methodName)) {
            return joinPoint.proceed();
        }

        String cacheKey = generateCacheKey(joinPoint);
        if (StringUtils.isEmpty(cacheKey)) {
            return joinPoint.proceed();
        }

        try {
            Object cachedValue = redisTemplate.opsForValue().get(cacheKey);
            if (cachedValue != null) {
                log.debug("从Redis缓存获取数据, method: {}, key: {}", methodName, cacheKey);
                return cachedValue;
            }
        } catch (Exception e) {
            log.warn("Redis缓存读取失败, method: {}, error: {}", methodName, e.getMessage());
        }

        Object result = joinPoint.proceed();

        if (result != null && isSuccessfulResponse(result)) {
            try {
                redisTemplate.opsForValue().set(cacheKey, result, CACHE_TTL_MINUTES, TimeUnit.MINUTES);
                log.debug("数据已缓存到Redis, method: {}, key: {}", methodName, cacheKey);
            } catch (Exception e) {
                log.warn("Redis缓存写入失败, method: {}, error: {}", methodName, e.getMessage());
            }
        }

        return result;
    }

    private boolean isCacheableMethod(String methodName) {
        return "findTransactionPath".equals(methodName)
                || "findAddressesWithinHops".equals(methodName)
                || "getAddressStats".equals(methodName)
                || "getAddressConnections".equals(methodName)
                || "analyzeAddressPattern".equals(methodName)
                || "getTransferStats".equals(methodName)
                || "findLargeTransfers".equals(methodName)
                || "findBTCHops".equals(methodName)
                || "findBTCPath".equals(methodName);
    }

    private String generateCacheKey(ProceedingJoinPoint joinPoint) {
        StringBuilder keyBuilder = new StringBuilder(CACHE_PREFIX);
        keyBuilder.append(joinPoint.getSignature().getName()).append(":");

        Object[] args = joinPoint.getArgs();
        if (args != null && args.length > 0) {
            for (int i = 0; i < args.length; i++) {
                if (args[i] != null) {
                    String argStr = args[i].toString();
                    if (argStr.length() > 100) {
                        argStr = argStr.substring(0, 100);
                    }
                    keyBuilder.append(argStr).append("_");
                }
            }
        }

        return keyBuilder.toString().replaceAll("[^a-zA-Z0-9:_]", "_");
    }

    private boolean isSuccessfulResponse(Object result) {
        if (result == null) {
            return false;
        }

        try {
            ObjectMapper mapper = new ObjectMapper();
            String json = mapper.writeValueAsString(result);
            return json.contains("\"code\":200") || json.contains("\"code\": 200");
        } catch (Exception e) {
            return true;
        }
    }
}