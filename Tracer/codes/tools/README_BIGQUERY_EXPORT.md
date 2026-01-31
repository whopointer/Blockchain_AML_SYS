# BigQuery 数据导出工具

## 功能说明

该工具用于从 BigQuery 查询代币转账数据，并按照格式2（简化CSV）输出为多个CSV文件，每个文件恰好包含 100000 个区块的数据。

## 安装依赖

首先需要添加 BigQuery Go 客户端库：

```bash
cd codes
go get cloud.google.com/go/bigquery
go mod tidy
```

## 配置认证

### 方式 1：使用服务账号密钥文件（推荐）

1. 在 Google Cloud Console 创建服务账号并下载 JSON 密钥文件
2. 设置环境变量：
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### 方式 2：使用 gcloud CLI

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

## 使用方法

### 基本用法

```bash
cd codes/tools
go run bigquery_export.go [输出目录]
```

示例：
```bash
go run bigquery_export.go ./output
```

### 编译后运行

```bash
cd codes/tools
go build -o bigquery_export bigquery_export.go
./bigquery_export ./output
```

## 输出文件格式

### 文件命名

文件按照区块范围命名：`{start_block}_{end_block}.csv`

例如：
- `17300000_17400000.csv` - 包含区块 17300000 到 17399999
- `17400000_17500000.csv` - 包含区块 17400000 到 17499999
- `17500000_17600000.csv` - 包含区块 17500000 到 17599999

### CSV 格式（格式2）

每行包含 8 个字段，用逗号分隔：

```csv
block,timestamp,index,hex(tx),string(coin),string(from),string(to),string_base10(value)
```

示例：
```csv
17300000,2023-05-20 12:34:56,0,0xabc123...,0xdAC17F958D2ee523a2206206994597C13D831ec7,0x1234...,0x5678...,1000000
```

**字段说明**：
- `block`: 区块号（整数）
- `timestamp`: 时间戳（格式：`YYYY-MM-DD HH:MM:SS`，UTC）
- `index`: 交易在区块中的索引（整数，如果为 NULL 则使用 0）
- `hex(tx)`: 交易哈希（十六进制字符串）
- `string(coin)`: 代币地址（十六进制字符串）
- `string(from)`: 发送地址（十六进制字符串）
- `string(to)`: 接收地址（十六进制字符串）
- `string_base10(value)`: 转账金额（十进制字符串）

## 自定义查询

如果需要修改查询条件，编辑 `bigquery_export.go` 中的 `queryAndExport` 函数：

```go
query := `
    SELECT 
        tt.block_number as block,
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', tt.block_timestamp) as timestamp,
        t.transaction_index as index,
        tt.transaction_hash as hex_tx,
        tt.token_address as coin,
        tt.from_address as from_addr,
        tt.to_address as to_addr,
        CAST(tt.value AS STRING) as value
    FROM 
        \`bigquery-public-data.crypto_ethereum.token_transfers\` tt
    LEFT JOIN 
        \`bigquery-public-data.crypto_ethereum.transactions\` t
        ON tt.transaction_hash = t.hash 
        AND tt.block_number = t.block_number
    WHERE 
        tt.block_number >= 17300000 AND tt.block_number < 17600000
        AND DATE(tt.block_timestamp) IN ('2023-05-20', '2023-07-01')
    ORDER BY 
        tt.block_number, t.transaction_index, tt.log_index
`
```

## 注意事项

1. **区块范围**：每个文件恰好包含 100000 个区块（BlockSpan）
2. **NULL 值处理**：如果 `transaction_index` 为 NULL，将使用 0 作为默认值
3. **文件顺序**：由于查询结果已按 `block_number` 排序，文件会按顺序创建
4. **内存使用**：程序使用流式处理，定期刷新文件，避免内存积累
5. **BigQuery 配额**：注意 BigQuery 的查询配额和费用限制

## 性能优化

- 查询结果已按区块号排序，便于分组
- 使用流式处理，避免一次性加载所有数据
- 定期刷新文件，减少内存占用
- 支持 BigQuery 查询缓存

## 故障排查

### 错误：认证失败

**解决**：
- 检查 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量
- 确认服务账号有 BigQuery 访问权限
- 运行 `gcloud auth application-default login`

### 错误：项目ID错误

**解决**：
- 设置 `GOOGLE_CLOUD_PROJECT` 环境变量
- 或在代码中直接修改 `projectID` 变量

### 错误：查询超时

**解决**：
- 减少查询的区块范围
- 增加更具体的 WHERE 条件
- 使用分区表优化查询

### 错误：磁盘空间不足

**解决**：
- 确保输出目录有足够的磁盘空间
- 考虑分批处理，减少每次查询的数据量

## 示例输出

```
开始执行 BigQuery 查询...
开始处理查询结果...
创建新文件: output/17300000_17400000.csv (BlockID: 173, 区块范围: 17300000-17400000)
已处理 10000 行数据 (批次 1)
已处理 20000 行数据 (批次 2)
...
创建新文件: output/17400000_17500000.csv (BlockID: 174, 区块范围: 17400000-17500000)
...
正在关闭所有文件...
文件已保存: 17300000_17400000.csv (BlockID: 173, 行数: 15234)
文件已保存: 17400000_17500000.csv (BlockID: 174, 行数: 18923)
总共处理了 34157 行数据，生成了 2 个文件
导出完成！
```
