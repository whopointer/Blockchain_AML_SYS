package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"time"

	"cloud.google.com/go/bigquery"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

const (
	BlockSpan = 100000 // 每个文件包含的区块数量
)

// BigQueryRow 表示查询结果的一行
type BigQueryRow struct {
	Block     int64  `bigquery:"block"`
	Timestamp string `bigquery:"timestamp"`
	Index     *int64 `bigquery:"index"` // 可能为 NULL
	HexTx     string `bigquery:"hex_tx"`
	Coin      string `bigquery:"coin"`
	FromAddr  string `bigquery:"from_addr"`
	ToAddr    string `bigquery:"to_addr"`
	Value     string `bigquery:"value"`
}

func main() {
	// 配置参数
	// 方式1：从环境变量读取（推荐）
	projectID := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if projectID == "" {
		fmt.Println("警告: GOOGLE_CLOUD_PROJECT 环境变量未设置")
		fmt.Println("请设置: export GOOGLE_CLOUD_PROJECT=your-project-id")
		projectID = "your-project-id" // 请替换为你的项目ID，或通过环境变量设置
	}

	credentialsPath := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	// credentialsPath 可以为空，将使用默认凭证（gcloud auth application-default login）

	outputDir := "output" // 默认输出目录
	if len(os.Args) > 1 {
		outputDir = os.Args[1]
	}
	fmt.Printf("项目ID: %s\n", projectID)
	fmt.Printf("输出目录: %s\n", outputDir)
	if credentialsPath != "" && credentialsPath != "path/to/credentials.json" {
		fmt.Printf("使用凭证文件: %s\n", credentialsPath)
	} else {
		fmt.Println("使用默认凭证（gcloud 或环境变量）")
	}

	// 创建输出目录
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Printf("创建输出目录失败: %v\n", err)
		os.Exit(1)
	}

	// 初始化 BigQuery 客户端
	ctx := context.Background()
	var client *bigquery.Client
	var err error

	if credentialsPath != "" && credentialsPath != "path/to/credentials.json" {
		// 使用凭证文件
		client, err = bigquery.NewClient(ctx, projectID, option.WithCredentialsFile(credentialsPath))
	} else {
		// 使用默认凭证（环境变量或 gcloud 配置）
		client, err = bigquery.NewClient(ctx, projectID)
	}
	if err != nil {
		fmt.Printf("创建 BigQuery 客户端失败: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// 执行查询并导出
	if err := queryAndExport(ctx, client, outputDir); err != nil {
		fmt.Printf("查询和导出失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("导出完成！")
}

// queryAndExport 执行查询并按区块范围分组导出
func queryAndExport(ctx context.Context, client *bigquery.Client, outputDir string) error {
	// SQL 查询语句
	// 注意：在 Go 的反引号字符串中，不能使用反引号，所以使用双引号字符串并转义
	query := "SELECT " +
		"tt.block_number as block, " +
		"FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', tt.block_timestamp) as timestamp, " +
		"t.transaction_index as index, " +
		"tt.transaction_hash as hex_tx, " +
		"tt.token_address as coin, " +
		"tt.from_address as from_addr, " +
		"tt.to_address as to_addr, " +
		"CAST(tt.value AS STRING) as value " +
		"FROM `bigquery-public-data.crypto_ethereum.token_transfers` tt " +
		"LEFT JOIN `bigquery-public-data.crypto_ethereum.transactions` t " +
		"ON tt.transaction_hash = t.hash " +
		"AND tt.block_number = t.block_number " +
		"WHERE tt.block_number >= 17300000 AND tt.block_number < 17600000 " +
		"AND DATE(tt.block_timestamp) IN ('2023-05-20', '2023-07-01') " +
		"ORDER BY tt.block_number, t.transaction_index, tt.log_index"

	fmt.Println("开始执行 BigQuery 查询...")
	fmt.Println("提示: BigQuery 查询可能需要几分钟时间，请耐心等待...")
	
	q := client.Query(query)
	q.DisableQueryCache = false // 允许使用查询缓存
	
	// 创建带超时的上下文（30分钟超时）
	queryCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	// 执行查询（这会启动查询作业，可能需要一些时间）
	fmt.Println("正在启动查询作业...")
	job, err := q.Run(queryCtx)
	if err != nil {
		return fmt.Errorf("启动查询失败: %w", err)
	}

	// 等待查询完成并显示进度
	fmt.Println("查询作业已启动，正在等待结果...")
	fmt.Printf("作业ID: %s\n", job.ID())
	fmt.Println("提示: 查询正在 BigQuery 服务器上执行，这可能需要几分钟...")
	fmt.Println("提示: 如果查询时间过长，可以在 BigQuery 控制台查看作业状态")
	
	// 等待查询完成
	status, err := job.Wait(queryCtx)
	if err != nil {
		return fmt.Errorf("查询执行失败: %w", err)
	}
	
	if err := status.Err(); err != nil {
		return fmt.Errorf("查询作业出错: %w", err)
	}
	
	fmt.Println("查询完成！")

	// 获取查询结果
	fmt.Println("查询完成，开始读取结果...")
	it, err := job.Read(queryCtx)
	if err != nil {
		return fmt.Errorf("读取查询结果失败: %w", err)
	}

	// 用于按区块范围分组的映射
	// 由于查询结果已按 block_number 排序，我们可以按顺序处理
	fileWriters := make(map[uint64]*csv.Writer) // key: blockID (block / BlockSpan)
	fileHandles := make(map[uint64]*os.File)
	rowCounts := make(map[uint64]int) // 记录每个文件的记录数

	rowCount := 0
	batchCount := 0
	const batchSize = 10000 // 每处理 batchSize 行输出一次进度

	fmt.Println("开始处理查询结果...")

	// 遍历查询结果
	for {
		var row BigQueryRow
		err := it.Next(&row)
		if err == iterator.Done {
			break
		}
		if err != nil {
			return fmt.Errorf("读取查询结果失败: %w", err)
		}

		// 计算当前行所属的 BlockID
		blockID := uint64(row.Block) / BlockSpan

		// 获取或创建对应 BlockID 的文件 writer
		writer, exists := fileWriters[blockID]
		if !exists {
			// 创建新文件
			startBlock := blockID * BlockSpan
			endBlock := startBlock + BlockSpan
			fileName := fmt.Sprintf("%s/%d_%d.csv", outputDir, startBlock, endBlock)

			file, err := os.Create(fileName)
			if err != nil {
				return fmt.Errorf("创建文件失败 %s: %w", fileName, err)
			}

			writer = csv.NewWriter(file)
			writer.Comma = ','

			// 格式2不需要标题行，直接写入数据
			// 格式2: block, timestamp, index, hex(tx), string(coin), string(from), string(to), string_base10(value)

			fileWriters[blockID] = writer
			fileHandles[blockID] = file
			rowCounts[blockID] = 0

			fmt.Printf("创建新文件: %s (BlockID: %d, 区块范围: %d-%d)\n", fileName, blockID, startBlock, endBlock)
		}

		// 处理 index 字段（可能为 NULL）
		indexValue := "0"
		if row.Index != nil {
			indexValue = strconv.FormatInt(*row.Index, 10)
		}

		// 准备 CSV 行数据（格式2：8个字段）
		csvRow := []string{
			strconv.FormatInt(row.Block, 10),        // block
			row.Timestamp,                           // timestamp
			indexValue,                              // index
			row.HexTx,                               // hex(tx)
			row.Coin,                                // string(coin)
			row.FromAddr,                            // string(from)
			row.ToAddr,                              // string(to)
			row.Value,                               // string_base10(value)
		}

		// 写入 CSV 行
		if err := writer.Write(csvRow); err != nil {
			return fmt.Errorf("写入 CSV 行失败: %w", err)
		}

		rowCount++
		rowCounts[blockID]++

		if rowCount%batchSize == 0 {
			// 定期刷新所有文件，避免内存积累
			for _, w := range fileWriters {
				w.Flush()
				if err := w.Error(); err != nil {
					return fmt.Errorf("刷新 CSV writer 失败: %w", err)
				}
			}
			batchCount++
			fmt.Printf("已处理 %d 行数据 (批次 %d)\n", rowCount, batchCount)
		}
	}

	// 刷新并关闭所有文件
	fmt.Println("正在关闭所有文件...")
	for blockID, writer := range fileWriters {
		writer.Flush()
		if err := writer.Error(); err != nil {
			return fmt.Errorf("刷新 CSV writer 失败 (BlockID: %d): %w", blockID, err)
		}
		if file := fileHandles[blockID]; file != nil {
			if err := file.Close(); err != nil {
				return fmt.Errorf("关闭文件失败 (BlockID: %d): %w", blockID, err)
			}
			startBlock := blockID * BlockSpan
			endBlock := startBlock + BlockSpan
			fmt.Printf("文件已保存: %d_%d.csv (BlockID: %d, 行数: %d)\n", startBlock, endBlock, blockID, rowCounts[blockID])
		}
	}

	fmt.Printf("总共处理了 %d 行数据，生成了 %d 个文件\n", rowCount, len(fileWriters))
	return nil
}
