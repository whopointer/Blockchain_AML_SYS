package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
	"transfer-graph-evm/experiment"
	"transfer-graph-evm/model"
	"transfer-graph-evm/utils"
)

func main() {
	// 定义命令行参数
	var (
		startBlockID      = flag.Uint("start_block", 0, "起始区块ID（BlockID，不是区块号）")
		endBlockID        = flag.Uint("end_block", 0, "结束区块ID（BlockID，不是区块号）")
		token             = flag.String("token", "USDT", "追踪的代币类型：USDT 或 ETH（默认：USDT）")
		srcAddressFile    = flag.String("src", "", "源地址文件路径（必需）")
		allowedAddressFile = flag.String("allowed", "", "允许的地址文件路径（可选，留空表示不限制）")
		forbiddenAddressFile = flag.String("forbidden", "", "禁止的地址文件路径（可选，留空表示不限制）")
		outputFile        = flag.String("output", "", "输出文件路径（不含扩展名，会自动生成 .topn.csv, .alln.csv, .alle.csv）")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "MFTracer - EVM 兼容区块链资金流追踪系统\n\n")
		fmt.Fprintf(os.Stderr, "用法: %s [选项]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "选项:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\n示例:\n")
		fmt.Fprintf(os.Stderr, "  %s -start_block=173 -end_block=175 -src=./addresses/src.txt -output=./results/trace_result\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -start_block=173 -end_block=175 -token=ETH -src=./addresses/src.txt -output=./results/trace_result\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -start_block=173 -end_block=175 -src=./addresses/src.txt -allowed=./addresses/allowed.txt -forbidden=./addresses/forbidden.txt -output=./results/trace_result\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "说明:\n")
		fmt.Fprintf(os.Stderr, "  - BlockID = 区块号 / 100000（例如：区块17300000对应BlockID=173）\n")
		fmt.Fprintf(os.Stderr, "  - 代币类型：USDT（默认）或 ETH\n")
		fmt.Fprintf(os.Stderr, "  - 源地址文件：每行一个地址（十六进制格式，如：0x1234...）\n")
		fmt.Fprintf(os.Stderr, "  - 输出文件会自动生成三个CSV文件：\n")
		fmt.Fprintf(os.Stderr, "    * {output}.topn.csv: Top N 节点结果\n")
		fmt.Fprintf(os.Stderr, "    * {output}.alln.csv: 所有节点结果\n")
		fmt.Fprintf(os.Stderr, "    * {output}.alle.csv: 所有边（交易）结果\n")
	}

	flag.Parse()

	// 验证必需参数
	if *startBlockID == 0 || *endBlockID == 0 {
		fmt.Fprintf(os.Stderr, "❌ 错误: 必须指定起始和结束区块ID\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if *srcAddressFile == "" {
		fmt.Fprintf(os.Stderr, "❌ 错误: 必须指定源地址文件路径\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if *outputFile == "" {
		fmt.Fprintf(os.Stderr, "❌ 错误: 必须指定输出文件路径\n\n")
		flag.Usage()
		os.Exit(1)
	}

	// 验证文件是否存在
	if _, err := os.Stat(*srcAddressFile); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "❌ 错误: 源地址文件不存在: %s\n", *srcAddressFile)
		os.Exit(1)
	}

	if *allowedAddressFile != "" {
		if _, err := os.Stat(*allowedAddressFile); os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "❌ 错误: 允许的地址文件不存在: %s\n", *allowedAddressFile)
			os.Exit(1)
		}
	}

	if *forbiddenAddressFile != "" {
		if _, err := os.Stat(*forbiddenAddressFile); os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "❌ 错误: 禁止的地址文件不存在: %s\n", *forbiddenAddressFile)
			os.Exit(1)
		}
	}

	// 初始化配置和日志系统
	model.InitGlobalTomlConfig()
	utils.SetupLoggers()

	// 解析代币类型（需要在创建日志文件之前，因为日志文件名包含 tokenName）
	var tokenAddress model.Address
	var tokenName string
	switch strings.ToUpper(*token) {
	case "USDT":
		tokenAddress = utils.USDTAddress
		tokenName = "USDT"
	case "ETH":
		tokenAddress = model.NativeTokenAddress
		tokenName = "ETH"
	default:
		fmt.Fprintf(os.Stderr, "❌ 错误: 不支持的代币类型 '%s'，仅支持 USDT 或 ETH\n", *token)
		os.Exit(1)
	}

	// 创建日志文件（包含所有输出）
	logFileName := fmt.Sprintf("./trace_%s_%d.log", tokenName, time.Now().Unix())
	logFile, err := os.Create(logFileName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 错误: 无法创建日志文件 %s: %v\n", logFileName, err)
		os.Exit(1)
	}
	defer logFile.Close()

	// 保存原始标准输出和标准错误
	originalStdout := os.Stdout
	originalStderr := os.Stderr

	// 为 stdout 创建管道
	stdoutReader, stdoutWriter, err := os.Pipe()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 错误: 无法创建 stdout 管道: %v\n", err)
		os.Exit(1)
	}
	defer stdoutReader.Close()
	defer stdoutWriter.Close()

	// 为 stderr 创建管道
	stderrReader, stderrWriter, err := os.Pipe()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 错误: 无法创建 stderr 管道: %v\n", err)
		os.Exit(1)
	}
	defer stderrReader.Close()
	defer stderrWriter.Close()

	// 启动 goroutine 从管道读取并同时写入控制台和日志文件
	go func() {
		io.Copy(io.MultiWriter(originalStdout, logFile), stdoutReader)
	}()
	go func() {
		io.Copy(io.MultiWriter(originalStderr, logFile), stderrReader)
	}()

	// 重定向标准输出和标准错误到管道写入端
	os.Stdout = stdoutWriter
	os.Stderr = stderrWriter

	// 确保在程序退出时恢复标准输出
	defer func() {
		os.Stdout = originalStdout
		os.Stderr = originalStderr
		stdoutWriter.Close()
		stderrWriter.Close()
	}()

	fmt.Printf("========================================\n")
	fmt.Printf("日志文件: %s\n", logFileName)
	fmt.Printf("========================================\n")

	// 显示配置信息
	fmt.Printf("========================================\n")
	fmt.Printf("MFTracer - 资金流追踪实验\n")
	fmt.Printf("========================================\n")
	fmt.Printf("起始区块ID: %d\n", *startBlockID)
	fmt.Printf("结束区块ID: %d\n", *endBlockID)
	fmt.Printf("追踪代币: %s (%s)\n", tokenName, tokenAddress.Hex())
	fmt.Printf("源地址文件: %s\n", *srcAddressFile)
	if *allowedAddressFile != "" {
		fmt.Printf("允许的地址文件: %s\n", *allowedAddressFile)
	} else {
		fmt.Printf("允许的地址文件: 无限制\n")
	}
	if *forbiddenAddressFile != "" {
		fmt.Printf("禁止的地址文件: %s\n", *forbiddenAddressFile)
	} else {
		fmt.Printf("禁止的地址文件: 无限制\n")
	}
	fmt.Printf("输出文件: %s\n", *outputFile)
	fmt.Printf("激活阈值: %.2f 美元\n", model.GlobalTomlConfig.Flow.ActivateThreshold)
	fmt.Printf("年龄限制: %d 跳\n", model.GlobalTomlConfig.Flow.AgeLimit)
	fmt.Printf("========================================\n\n")

	// 转换参数类型
	sBlockID := uint16(*startBlockID)
	eBlockID := uint16(*endBlockID)

	// 处理可选参数（空字符串转换为空字符串，保持兼容）
	allowedFile := *allowedAddressFile
	forbiddenFile := *forbiddenAddressFile

	// 创建上下文
	ctx := context.Background()

	// 调用追踪函数
	fmt.Printf("开始追踪资金流...\n")
	err = experiment.TraceDownstream(
		ctx,
		sBlockID,
		eBlockID,
		tokenAddress,
		*srcAddressFile,
		allowedFile,
		forbiddenFile,
		*outputFile,
	)

	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 追踪失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n========================================\n")
	fmt.Printf("✅ 追踪完成！\n")
	fmt.Printf("========================================\n")
	fmt.Printf("输出文件:\n")
	fmt.Printf("  - %s.topn.csv (Top N 节点)\n", *outputFile)
	fmt.Printf("  - %s.alln.csv (所有节点)\n", *outputFile)
	fmt.Printf("  - %s.alle.csv (所有边/交易)\n", *outputFile)
	fmt.Printf("日志文件:\n")
	fmt.Printf("  - %s (所有调试输出)\n", logFileName)
	fmt.Printf("========================================\n")
}
