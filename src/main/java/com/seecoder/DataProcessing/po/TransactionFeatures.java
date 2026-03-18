// com/seecoder/DataProcessing/po/TransactionFeatures.java
package com.seecoder.DataProcessing.po;

import lombok.Data;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Data
public class TransactionFeatures {
    // 基础特征
    private Double txFee;               // 已取log
    private Long txSizeBytes;
    private Double feeRate;
    private Integer numInputs;
    private Integer numOutputs;
    private Double totalInputAmt;       // 已取log
    private Double totalOutputAmt;      // 已取log
    private Double outputStdDev;
    private Long locktime;
    private Integer isCoinbase;          // 0或1

    // 邻居特征
    private Double neighborInAmtMean;    // 已取log
    private Double neighborInAmtStd;     // 已取log
    private Double neighborInFeeMean;    // 已取log
    private Double neighborInDegreeMean;
    private Double neighborOutDegreeMean;
    private Double neighborTimeSpan;     // 平均时间差（秒）

    // 高级特征
    private Double coindaysDestroyed;    // 币天销毁（可考虑取log，按指南不要求）
    private Double roundValueRatio;      // 0~1之间的比例
    private Integer addressReuseCount;   // 最大复用次数
    private Integer loopFlag;            // 0或1
    private Double fanInRatio;

    // 将特征转换为CSV行（按固定顺序）
    public String[] toCsvRow() {
        List<String> fields = Stream.of(
                String.valueOf(txFee),
                String.valueOf(txSizeBytes),
                String.valueOf(feeRate),
                String.valueOf(numInputs),
                String.valueOf(numOutputs),
                String.valueOf(totalInputAmt),
                String.valueOf(totalOutputAmt),
                String.valueOf(outputStdDev),
                String.valueOf(locktime),
                String.valueOf(isCoinbase),
                String.valueOf(neighborInAmtMean),
                String.valueOf(neighborInAmtStd),
                String.valueOf(neighborInFeeMean),
                String.valueOf(neighborInDegreeMean),
                String.valueOf(neighborOutDegreeMean),
                String.valueOf(neighborTimeSpan),
                String.valueOf(coindaysDestroyed),
                String.valueOf(roundValueRatio),
                String.valueOf(addressReuseCount),
                String.valueOf(loopFlag),
                String.valueOf(fanInRatio)
        ).collect(Collectors.toList());
        return fields.toArray(new String[0]);
    }

    // 返回CSV表头（与toCsvRow顺序一致）
    public static String[] getCsvHeader() {
        return new String[]{
                "tx_fee", "tx_size_bytes", "fee_rate", "num_inputs", "num_outputs",
                "total_input_amt", "total_output_amt", "output_std_dev", "locktime", "is_coinbase",
                "neighbor_in_amt_mean", "neighbor_in_amt_std", "neighbor_in_fee_mean",
                "neighbor_in_degree_mean", "neighbor_out_degree_mean", "neighbor_time_span",
                "coindays_destroyed", "round_value_ratio", "address_reuse_count", "loop_flag",
                "fan_in_ratio"
        };
    }
}