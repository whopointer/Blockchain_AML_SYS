package com.seecoder.DataProcessing.util;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * 图形数据格式化工具类
 */
public class GraphFormatUtils {
    
    /**
     * 缩短地址显示，格式为前5位+...+后5位
     * @param address 完整地址
     * @return 缩短后的地址
     */
    public static String shortenAddress(String address) {
        if (address == null || address.length() <= 8) {
            return address;
        }
        return address.substring(0, 5) + "..." + address.substring(address.length() - 5);
    }
    
    /**
     * 格式化金额标签，根据链类型和金额大小选择不同的显示格式
     * @param amount 金额
     * @param chain 区块链类型
     * @return 格式化后的金额标签
     */
    public static String formatAmountLabel(double amount, String chain) {
        String unit = (chain != null && chain.equalsIgnoreCase("ethereum")) ? "ETH" : "BNB";
        if (amount >= 1.0) {
            return String.format("%.3f %s", amount, unit);
        } else {
            return String.format("%.6f %s", amount, unit);
        }
    }
    
    /**
     * 解析时间戳并转换为UTC时间字符串
     * @param timestampStr 时间戳字符串，可以是数字格式或多种字符串格式
     * @return 格式化后的UTC时间字符串，解析失败返回原始字符串
     */
    public static String parseAndFormatTimestamp(String timestampStr) {
        if (timestampStr == null || timestampStr.isEmpty()) {
            return timestampStr;
        }
        
        try {
            if (timestampStr.matches("-?\\d+(\\.\\d+)?([Ee][+-]?\\d+)?")) {
                double timestampDouble = Double.parseDouble(timestampStr);
                long milliseconds = (long) (timestampDouble * 1000);
                LocalDateTime utcDateTime = LocalDateTime.ofInstant(Instant.ofEpochMilli(milliseconds), ZoneId.of("UTC"));
                return utcDateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            } else {
                LocalDateTime dateTime;
                
                String[] formats = {
                        "yyyy-MM-dd HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss",
                        "yyyy-MM-dd'T'HH:mm:ss.SSS",
                        "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
                };
                
                for (String format : formats) {
                    try {
                        DateTimeFormatter formatter = DateTimeFormatter.ofPattern(format);
                        dateTime = LocalDateTime.parse(timestampStr, formatter);
                        return dateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                    } catch (Exception e) {
                    }
                }
                
                dateTime = LocalDateTime.parse(timestampStr, DateTimeFormatter.ISO_DATE_TIME);
                return dateTime.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            }
        } catch (Exception e) {
            return timestampStr; // 解析失败时返回原始字符串
        }
    }
}