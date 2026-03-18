// com/seecoder/DataProcessing/util/FeatureUtils.java
package com.seecoder.DataProcessing.util;

import com.seecoder.DataProcessing.po.ChainTx;
import com.seecoder.DataProcessing.po.ChainTxInput;
import com.seecoder.DataProcessing.po.ChainTxOutput;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

public class FeatureUtils {

    public static Double logTransform(Double value) {
        if (value == null) return null;
        return Math.log(value + 1e-8);
    }

    public static Double stdDev(List<Double> values) {
        if (values == null || values.isEmpty()) return 0.0;
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = values.stream().mapToDouble(v -> Math.pow(v - mean, 2)).average().orElse(0.0);
        return Math.sqrt(variance);
    }

    public static Double mean(List<ChainTx> txs, ToDoubleFunction<ChainTx> extractor) {
        if (txs == null || txs.isEmpty()) return 0.0;
        return txs.stream().mapToDouble(extractor).average().orElse(0.0);
    }

    public static Double stdDev(List<ChainTx> txs, ToDoubleFunction<ChainTx> extractor) {
        if (txs == null || txs.isEmpty()) return 0.0;
        double mean = mean(txs, extractor);
        double variance = txs.stream().mapToDouble(t -> Math.pow(extractor.applyAsDouble(t) - mean, 2)).average().orElse(0.0);
        return Math.sqrt(variance);
    }

    public static Double meanTimeSpan(LocalDateTime currentTime, List<LocalDateTime> prevTimes) {
        if (prevTimes == null || prevTimes.isEmpty()) return 0.0;
        return prevTimes.stream()
                .mapToLong(t -> ChronoUnit.SECONDS.between(t, currentTime))
                .average()
                .orElse(0.0);
    }

    public static Double outputStdDev(List<ChainTxOutput> outputs) {
        List<Double> values = outputs.stream()
                .map(o -> o.getValue() == null ? 0.0 : o.getValue().doubleValue())
                .collect(Collectors.toList());
        return stdDev(values);
    }

    /**
     * 计算输出金额中整数金额的比例（按个数比例）
     * @param outputs 输出列表
     * @return 整数输出个数 / 总输出个数，若列表为空返回0.0
     */
    public static Double roundValueRatio(List<ChainTxOutput> outputs) {
        if (outputs == null || outputs.isEmpty()) return 0.0;
        long roundCount = outputs.stream()
                .filter(o -> o.getValue() != null && o.getValue().remainder(BigDecimal.ONE).compareTo(BigDecimal.ZERO) == 0)
                .count();
        return (double) roundCount / outputs.size();
    }
}