package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"transfer-graph-evm/data"
	"transfer-graph-evm/graph"
	"transfer-graph-evm/model"
	"transfer-graph-evm/pricedb"
	"transfer-graph-evm/utils"
)

const (
	// BlockSpan 每个文件包含的区块数量（100000个区块）
	BlockSpan = 100000
)

// CSVFileInfo CSV文件信息结构体
type CSVFileInfo struct {
	FilePath  string  // CSV文件路径
	StartBlock uint64  // 起始区块号
	EndBlock   uint64  // 结束区块号
	BlockID    uint16  // 区块ID（BlockID = StartBlock / BlockSpan）
}

// parseCSVFileName 从文件名解析区块范围信息
// 文件名格式：{startBlock}_{endBlock}.csv
// 例如：17300000_17400000.csv
func parseCSVFileName(filePath string) (CSVFileInfo, error) {
	fileName := filepath.Base(filePath)  // 获取文件名（不含路径）
	
	// 使用正则表达式匹配文件名格式：数字_数字.csv
	re := regexp.MustCompile(`^(\d+)_(\d+)\.csv$`)
	matches := re.FindStringSubmatch(fileName)
	if len(matches) != 3 {
		return CSVFileInfo{}, fmt.Errorf("文件名格式不正确，应为：{startBlock}_{endBlock}.csv，实际：%s", fileName)
	}

	startBlock, err := strconv.ParseUint(matches[1], 10, 64)  // 解析起始区块号
	if err != nil {
		return CSVFileInfo{}, fmt.Errorf("解析起始区块号失败: %w", err)
	}

	endBlock, err := strconv.ParseUint(matches[2], 10, 64)  // 解析结束区块号
	if err != nil {
		return CSVFileInfo{}, fmt.Errorf("解析结束区块号失败: %w", err)
	}

	// 验证区块范围
	if endBlock <= startBlock {
		return CSVFileInfo{}, fmt.Errorf("结束区块号必须大于起始区块号")
	}

	if endBlock-startBlock != BlockSpan {
		return CSVFileInfo{}, fmt.Errorf("区块范围必须是 %d 个区块", BlockSpan)
	}

	// 计算 BlockID
	blockID := uint16(startBlock / BlockSpan)

	return CSVFileInfo{
		FilePath:  filePath,
		StartBlock: startBlock,
		EndBlock:   endBlock,
		BlockID:    blockID,
	}, nil
}

// readCSVFile 读取CSV文件并解析为 Transfer 和 Tx 列表
// 参数：
//   - file: CSV文件
//   - isNativeTokenFile: 是否为原生代币交易文件（true=原生代币，false=代币交易）
// 自动检测CSV格式：
//   - CSV版本2（8字段）：根据isNativeTokenFile参数决定处理方式
//     - isNativeTokenFile=false: 代币交易 → Transfer
//     - isNativeTokenFile=true: 原生代币交易 → Transfer（临时）→ Tx（方案1）
//   - CSV版本4（11字段）：原生代币交易 → Tx（直接解析）
func readCSVFile(file *os.File, isNativeTokenFile bool) ([]*model.Transfer, []*model.Tx, error) {
	var transfers []*model.Transfer  // 代币转账列表
	var txs []*model.Tx              // 原生代币交易列表

	scanner := bufio.NewScanner(file)  // 创建文件扫描器
	lineNum := 0  // 行号计数器
	var csvVersion int  // CSV版本（2或4），首次检测后确定

	for scanner.Scan() {  // 逐行扫描文件
		lineNum++  // 递增行号
		line := strings.TrimSpace(scanner.Text())  // 去除首尾空白字符

		// 跳过空行
		if line == "" {
			continue
		}

		// 如果是第一行，尝试检测CSV格式
		if lineNum == 1 {
			// 检查是否是标题行
			if strings.Contains(line, "block") || strings.Contains(line, "timestamp") {
				continue  // 跳过标题行
			}
			
			// 通过字段数量检测CSV版本
			reader := csv.NewReader(strings.NewReader(line))
			parts, err := reader.Read()
			if err != nil {
				return nil, nil, fmt.Errorf("第 %d 行解析失败: %w\n行内容: %s", lineNum, err, line)
			}
			
			switch len(parts) {
			case 8:
				csvVersion = 2  // CSV版本2：8字段格式
			default:
				return nil, nil, fmt.Errorf("不支持的CSV格式：第 %d 行有 %d 个字段（仅支持8字段的CSV版本2格式）\n行内容: %s", lineNum, len(parts), line)
			}
		}

		// 根据检测到的CSV版本和文件类型解析
		// 注意：当前实现只支持CSV版本2格式
		// CSV版本4的支持已移除，因为当前需求只需要处理CSV版本2格式的两个文件
		if csvVersion == 2 && isNativeTokenFile {
			// CSV版本2 + 原生代币文件：同时解析为Transfer和Tx
			// 1. 解析为Transfer（用于qres.Transfers，确保ConstructSubgraphs能创建原生代币子图）
			// 2. 转换为Tx（用于qres.Txs，存储原生代币交易数据）
			csvLine := data.Res_Transaction_CSVLine_Ver2(line)  // 转换为CSV版本2类型
			transfer, err := csvLine.Decode()  // 调用Decode方法解析为Transfer
			if err != nil {
				return nil, nil, fmt.Errorf("第 %d 行解析失败: %w\n行内容: %s", lineNum, err, line)
			}
			
			// 验证是否为原生代币交易（coin字段应为0x0000000000000000000000000000000000000000）
			if !model.IsNativeToken(transfer.Token) {
				return nil, nil, fmt.Errorf("第 %d 行：原生代币交易文件的coin字段应为0x0000000000000000000000000000000000000000，实际为: %s\n行内容: %s", lineNum, transfer.Token.Hex(), line)
			}
			
			// 添加到Transfer列表（用于qres.Transfers）
			transfers = append(transfers, transfer)
			
			// 转换为 model.Tx 对象并添加到Tx列表（用于qres.Txs）
			tx := &model.Tx{
				Block:      transfer.Block(),
				Time:       transfer.Timestamp,
				Index:      transfer.Index(),
				TxHash:     transfer.TxHash,
				From:       transfer.From,
				To:         transfer.To,
				IsCreation: false,  // CSV版本2无此信息
				Value:      transfer.Value,
				Fee:        nil,    // CSV版本2无此信息
				Func:       "",     // CSV版本2无此信息
				Param:      nil,    // CSV版本2无此信息
			}
			txs = append(txs, tx)  // 添加到原生代币交易列表
		} else {
			// CSV版本2 + 代币交易文件：解析为 Transfer
			csvLine := data.Res_Transaction_CSVLine_Ver2(line)  // 转换为CSV版本2类型
			transfer, err := csvLine.Decode()  // 调用Decode方法解析为Transfer
			if err != nil {
				return nil, nil, fmt.Errorf("第 %d 行解析失败: %w\n行内容: %s", lineNum, err, line)
			}
			
			// 验证是否为代币交易（coin字段不应为0x0000000000000000000000000000000000000000）
			if model.IsNativeToken(transfer.Token) {
				return nil, nil, fmt.Errorf("第 %d 行：代币交易文件的coin字段不应为0x0000000000000000000000000000000000000000\n行内容: %s", lineNum, line)
			}
			
			transfers = append(transfers, transfer)  // 添加到代币转账列表
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, nil, fmt.Errorf("读取文件失败: %w", err)
	}

	return transfers, txs, nil
}

// extractUniqueTokens 从转账列表中提取所有唯一的代币地址
func extractUniqueTokens(transfers []*model.Transfer) map[string]struct{} {
	tokenSet := make(map[string]struct{})  // 创建代币集合
	for _, t := range transfers {  // 遍历转账列表
		tokenStr := string(t.Token.Bytes())  // 将代币地址转换为字符串
		tokenSet[tokenStr] = struct{}{}  // 添加到集合
	}
	return tokenSet  // 返回代币集合
}

// processCSVFile 处理单个CSV文件，返回解析结果
// 参数：
//   - isNativeTokenFile: 是否为原生代币交易文件
// 返回值：
//   - transfers: 代币转账列表（仅当isNativeTokenFile=false时有效）
//   - txs: 原生代币交易列表（仅当isNativeTokenFile=true时有效）
func processCSVFile(ctx context.Context, fileInfo CSVFileInfo, isNativeTokenFile bool) ([]*model.Transfer, []*model.Tx, error) {
	// 打开CSV文件
	file, err := os.Open(fileInfo.FilePath)
	if err != nil {
		return nil, nil, fmt.Errorf("打开文件失败: %w", err)
	}
	defer file.Close()

	// 读取并解析CSV
	transfers, txs, err := readCSVFile(file, isNativeTokenFile)
	if err != nil {
		return nil, nil, fmt.Errorf("读取CSV文件失败: %w", err)
	}

	if len(transfers) == 0 && len(txs) == 0 {
		return nil, nil, fmt.Errorf("CSV文件中没有数据")
	}

	if isNativeTokenFile {
		fmt.Printf("  读取到 %d 条原生代币转账记录（Transfer）和 %d 条原生代币交易记录（Tx）\n", len(transfers), len(txs))
	} else {
		fmt.Printf("  读取到 %d 条代币转账记录\n", len(transfers))
	}

	return transfers, txs, nil
}

// writeToGraphDB 将合并后的数据写入图数据库
func writeToGraphDB(ctx context.Context, transfers []*model.Transfer, txs []*model.Tx, blockID uint16, g *graph.GraphDB, pdb *pricedb.PriceDB) error {
	startTime := time.Now()  // 记录开始时间

	if len(transfers) == 0 && len(txs) == 0 {
		return fmt.Errorf("没有数据需要写入")
	}

	// 提取所有唯一的代币地址并添加到支持列表（只处理代币转账，不包括原生代币）
	tokenSet := extractUniqueTokens(transfers)
	fmt.Printf("  发现 %d 个唯一代币\n", len(tokenSet))

	// 确保 SupportTokenMap 已初始化
	if model.SupportTokenMap == nil {
		model.SupportTokenMap = make(map[string]struct{})
	}

	var newTokensAdded int  // 新添加的代币数量
	var tokenList []model.Address  // 临时列表用于收集新代币

	// 添加所有代币到支持列表
	for tokenStr := range tokenSet {
		token := model.BytesToAddress([]byte(tokenStr))
		// 检查是否已经在支持列表中
		if !model.IsSupportToken(token) {
			model.SupportTokenMap[tokenStr] = struct{}{}
			tokenList = append(tokenList, token)
			newTokensAdded++
		}
	}

	// 更新 SupportTokenList
	if len(tokenList) > 0 {
		model.SupportTokenList = append(model.SupportTokenList, tokenList...)
		fmt.Printf("  已将 %d 个新代币添加到支持列表\n", newTokensAdded)
	} else {
		fmt.Printf("  所有代币已在支持列表中\n")
	}

	// 构建 QueryResult
	// 注意：文件2（原生代币交易）已经同时解析为Transfer和Tx
	// 因此 qres.Transfers 中已经包含了原生代币地址的转账记录，ConstructSubgraphs 可以正常创建子图
	if len(txs) > 0 {
		// 确保原生代币地址在支持列表中（这样原生代币转账记录不会被filterTss过滤掉）
		nativeTokenAddressStr := string(model.NativeTokenAddress.Bytes())
		if !model.IsSupportToken(model.NativeTokenAddress) {
			model.SupportTokenMap[nativeTokenAddressStr] = struct{}{}
			model.SupportTokenList = append(model.SupportTokenList, model.NativeTokenAddress)
			fmt.Printf("  已将原生代币地址添加到支持列表\n")
		}
	}
	
	qres := &model.QueryResult{
		Transfers: transfers,  // 代币转账列表（包含文件1的代币转账和文件2的原生代币转账）
		Txs:       txs,        // 原生代币交易列表（仅来自文件2）
	}
	
	fmt.Printf("  构建 QueryResult 完成: %d 条转账记录（Transfer）, %d 条原生代币交易（Tx）\n", len(transfers), len(txs))

	// 写入图数据库
	fmt.Printf("  正在写入图数据库...\n")
	err := graph.SyncFromQresSimple(ctx, qres, blockID, g, pdb, 10)  // 并行度设为10
	if err != nil {
		return fmt.Errorf("写入图数据库失败: %w", err)
	}

	duration := time.Since(startTime)  // 计算耗时
	fmt.Printf("  ✅ 写入成功，耗时: %v\n", duration)
	return nil
}

// findCSVFiles 查找指定路径下的所有CSV文件
func findCSVFiles(inputPath string) ([]string, error) {
	var csvFiles []string  // CSV文件列表

	// 检查输入路径是文件还是目录
	info, err := os.Stat(inputPath)
	if err != nil {
		return nil, fmt.Errorf("无法访问路径: %w", err)
	}

	if info.IsDir() {
		// 如果是目录，查找所有CSV文件
		err := filepath.Walk(inputPath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".csv") {
				csvFiles = append(csvFiles, path)
			}
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("遍历目录失败: %w", err)
		}
	} else {
		// 如果是文件，直接添加
		if strings.HasSuffix(strings.ToLower(inputPath), ".csv") {
			csvFiles = append(csvFiles, inputPath)
		} else {
			return nil, fmt.Errorf("输入文件不是CSV格式: %s", inputPath)
		}
	}

	if len(csvFiles) == 0 {
		return nil, fmt.Errorf("未找到CSV文件")
	}

	return csvFiles, nil
}

func main() {
	// 检查命令行参数
	if len(os.Args) < 3 {
		fmt.Println("用法: csv_import <代币交易CSV文件路径> <原生代币交易CSV文件路径> [图数据库路径] [价格数据库路径]")
		fmt.Println("")
		fmt.Println("示例:")
		fmt.Println("  ./csv_import ./output/token_17300000_17400000.csv ./output/native_17300000_17400000.csv")
		fmt.Println("  ./csv_import ./output/token_17300000_17400000.csv ./output/native_17300000_17400000.csv ./graph_db ./price_db")
		fmt.Println("")
		fmt.Println("注意: 如果可执行文件在当前目录，需要使用 ./csv_import 来运行")
		fmt.Println("")
		fmt.Println("说明:")
		fmt.Println("  - 第一个参数：代币交易CSV文件（CSV版本2格式，coin字段为代币地址）")
		fmt.Println("  - 第二个参数：原生代币交易CSV文件（CSV版本2格式，coin字段为0x0000000000000000000000000000000000000000）")
		fmt.Println("  - 两个CSV文件必须包含相同的区块范围")
		fmt.Println("  - CSV文件命名格式：{startBlock}_{endBlock}.csv（例如：17300000_17400000.csv）")
		fmt.Println("  - 每个CSV文件必须包含恰好100000个区块的数据")
		fmt.Println("  - 如果未指定数据库路径，将从config.toml读取")
		os.Exit(1)
	}

	tokenCSVPath := os.Args[1]    // 代币交易CSV文件路径
	nativeCSVPath := os.Args[2]   // 原生代币交易CSV文件路径

	// 初始化配置和日志系统
	model.InitGlobalTomlConfig()  // 初始化全局TOML配置
	utils.SetupLoggers()  // 初始化日志系统（必须在调用任何使用日志的函数之前）

	// 确定数据库路径
	var graphDBPath, priceDBPath string
	if len(os.Args) >= 4 {
		graphDBPath = os.Args[3]  // 从命令行参数获取图数据库路径
	} else {
		graphDBPath = model.GetConfigDBPath()  // 从配置文件获取图数据库路径
	}

	if len(os.Args) >= 5 {
		priceDBPath = os.Args[4]  // 从命令行参数获取价格数据库路径
	} else {
		priceDBPath = model.GetConfigPriceDBPath()  // 从配置文件获取价格数据库路径
	}

	fmt.Printf("代币交易CSV文件: %s\n", tokenCSVPath)
	fmt.Printf("原生代币交易CSV文件: %s\n", nativeCSVPath)
	fmt.Printf("图数据库路径: %s\n", graphDBPath)
	fmt.Printf("价格数据库路径: %s\n", priceDBPath)

	// 验证CSV文件是否存在
	if _, err := os.Stat(tokenCSVPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "❌ 代币交易CSV文件不存在: %s\n", tokenCSVPath)
		os.Exit(1)
	}
	if _, err := os.Stat(nativeCSVPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "❌ 原生代币交易CSV文件不存在: %s\n", nativeCSVPath)
		os.Exit(1)
	}

	// 初始化图数据库
	fmt.Printf("\n正在初始化图数据库...\n")
	g, err := graph.NewGraphDB(graphDBPath, false)  // false表示非只读模式
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 初始化图数据库失败: %v\n", err)
		os.Exit(1)
	}
	defer g.Close()  // 确保关闭数据库
	fmt.Printf("图数据库初始化成功\n")

	// 初始化价格数据库
	fmt.Printf("正在初始化价格数据库...\n")
	pdb, err := pricedb.NewPriceDB(priceDBPath, false)  // false表示非只读模式
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 初始化价格数据库失败: %v\n", err)
		os.Exit(1)
	}
	defer pdb.Close()  // 确保关闭数据库
	fmt.Printf("价格数据库初始化成功\n")

	// 创建上下文
	ctx := context.Background()

	// 解析两个CSV文件的文件名，验证区块范围是否一致
	tokenFileInfo, err := parseCSVFileName(tokenCSVPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 解析代币交易CSV文件名失败: %v\n", err)
		os.Exit(1)
	}

	nativeFileInfo, err := parseCSVFileName(nativeCSVPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 解析原生代币交易CSV文件名失败: %v\n", err)
		os.Exit(1)
	}

	// 验证两个文件的区块范围是否一致
	if tokenFileInfo.StartBlock != nativeFileInfo.StartBlock || tokenFileInfo.EndBlock != nativeFileInfo.EndBlock {
		fmt.Fprintf(os.Stderr, "❌ 两个CSV文件的区块范围不一致:\n")
		fmt.Fprintf(os.Stderr, "  代币交易文件: %d-%d\n", tokenFileInfo.StartBlock, tokenFileInfo.EndBlock)
		fmt.Fprintf(os.Stderr, "  原生代币交易文件: %d-%d\n", nativeFileInfo.StartBlock, nativeFileInfo.EndBlock)
		os.Exit(1)
	}

	if tokenFileInfo.BlockID != nativeFileInfo.BlockID {
		fmt.Fprintf(os.Stderr, "❌ 两个CSV文件的BlockID不一致:\n")
		fmt.Fprintf(os.Stderr, "  代币交易文件: BlockID=%d\n", tokenFileInfo.BlockID)
		fmt.Fprintf(os.Stderr, "  原生代币交易文件: BlockID=%d\n", nativeFileInfo.BlockID)
		os.Exit(1)
	}

	fmt.Printf("\n区块范围: %d-%d, BlockID: %d\n", tokenFileInfo.StartBlock, tokenFileInfo.EndBlock, tokenFileInfo.BlockID)

	// 处理代币交易CSV文件
	fmt.Printf("\n[1/2] 处理代币交易文件: %s\n", filepath.Base(tokenCSVPath))
	tokenTransfers, _, err := processCSVFile(ctx, tokenFileInfo, false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 处理代币交易文件失败: %v\n", err)
		os.Exit(1)
	}

	// 处理原生代币交易CSV文件
	fmt.Printf("\n[2/2] 处理原生代币交易文件: %s\n", filepath.Base(nativeCSVPath))
	nativeTransfers, nativeTxs, err := processCSVFile(ctx, nativeFileInfo, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 处理原生代币交易文件失败: %v\n", err)
		os.Exit(1)
	}

	// 合并两个文件的转账记录
	allTransfers := append(tokenTransfers, nativeTransfers...)
	fmt.Printf("  合并后共有 %d 条转账记录（包含 %d 条代币转账和 %d 条原生代币转账）\n", 
		len(allTransfers), len(tokenTransfers), len(nativeTransfers))

	// 合并结果并写入图数据库
	fmt.Printf("\n[3/3] 合并结果并写入图数据库...\n")
	err = writeToGraphDB(ctx, allTransfers, nativeTxs, tokenFileInfo.BlockID, g, pdb)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 写入图数据库失败: %v\n", err)
		os.Exit(1)
	}

	// 输出处理结果
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("✅ 处理完成！两个CSV文件已成功导入图数据库\n")
	fmt.Printf("%s\n", strings.Repeat("=", 60))
}
