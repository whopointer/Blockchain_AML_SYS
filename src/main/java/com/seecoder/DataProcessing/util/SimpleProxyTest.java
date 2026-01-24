package com.seecoder.DataProcessing.util;

import java.net.*;
import java.io.*;

public class SimpleProxyTest {
    public static void main(String[] args) {
        // 使用与BigQueryConfig中相同的代理设置
        String proxyHost = "127.0.0.1";
        int proxyPort = 7897;
        String targetUrl = "https://www.googleapis.com"; // Google API 域名

        System.setProperty("https.proxyHost", proxyHost);
        System.setProperty("https.proxyPort", String.valueOf(proxyPort));

        try {
            URL url = new URL(targetUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);

            int responseCode = conn.getResponseCode();
            System.out.println("✅ 网络测试通过！连接到 " + targetUrl + " 的响应码: " + responseCode);
        } catch (SocketTimeoutException e) {
            System.err.println("❌ 连接超时。请检查：");
            System.err.println("   1. 代理地址/端口 (" + proxyHost + ":" + proxyPort + ") 是否正确？");
            System.err.println("   2. 代理服务是否正在运行？");
            System.err.println("   3. 防火墙是否放行？");
        } catch (IOException e) {
            System.err.println("❌ 连接失败: " + e.getMessage());
        }
    }
}