# CSV导入工具使用说明

## 概述

`csv_import.go` 是一个命令行工具，用于将CSV版本2格式的文件导入到图数据库中。该工具使用 `codes/data` 和 `codes/encoding` 模块提供的功能来解析CSV文件并写入图数据库。

## 功能特性

- ✅ 支持CSV版本2格式（8字段格式）
- ✅ 自动解析文件名获取区块范围信息
- ✅ 支持单个文件或目录批量导入
- ✅ 自动提取并添加代币到支持列表
- ✅ 使用 `graph.SyncFromQresSimple` 写入图数据库
- ✅ 详细的进度和错误信息输出

## CSV文件格式要求

### 文件命名格式

CSV文件必须按照以下格式命名：
```
{startBlock}_{endBlock}.csv
```

例如：
- `17300000_17400000.csv` - 包含区块17300000到17400000的数据
- `17400000_17500000.csv` - 包含区块17400000到17500000的数据

**重要**：每个CSV文件必须包含恰好100000个区块的数据（BlockSpan = 100000）。

### CSV内容格式（版本2）

CSV文件应包含以下8个字段（逗号分隔，无标题行）：

```csv
block,timestamp,index,hex(tx),string(coin),string(from),string(to),string_base10(value)
```

**字段说明**：
- `block`: 区块号（整数）
- `timestamp`: 时间戳（格式：`"YYYY-MM-DD HH:MM:SS"`，UTC时区）
- `index`: 交易索引（整数）
- `hex(tx)`: 交易哈希（十六进制字符串）
- `string(coin)`: 代币地址（十六进制字符串）
- `string(from)`: 发送地址（十六进制字符串）
- `string(to)`: 接收地址（十六进制字符串）
- `string_base10(value)`: 转账金额（十进制字符串）

**示例**：
```csv
18000000,"2025-12-12 01:48:12",0,0xef01...,0xdAC17F958D2ee523a2206206994597C13D831ec7,0x1234...,0x5678...,1000000
```

## 编译

```bash
cd codes/tools
go build -o csv_import csv_import.go
```

## 使用方法

### 基本用法

```bash
# 导入单个CSV文件（使用config.toml中的数据库路径）
./csv_import ./output/17400000_17500000.csv

# 导入目录中的所有CSV文件
./csv_import ./output/

# 指定自定义数据库路径
./csv_import ./output/17400000_17500000.csv ./graph_db ./price_db
```

### 命令行参数

```
csv_import <CSV文件路径或目录> [图数据库路径] [价格数据库路径]
```

- `CSV文件路径或目录`（必需）：要导入的CSV文件路径或包含CSV文件的目录
- `图数据库路径`（可选）：图数据库存储路径，如果未指定则从 `config.toml` 读取
- `价格数据库路径`（可选）：价格数据库存储路径，如果未指定则从 `config.toml` 读取

## 工作流程

1. **解析文件名**：从文件名提取区块范围信息（startBlock, endBlock, BlockID）
2. **读取CSV文件**：逐行读取CSV文件
3. **解析CSV行**：使用 `data.Res_Transaction_CSVLine_Ver2.Decode()` 解析每行
4. **提取代币**：自动提取所有唯一的代币地址并添加到 `model.SupportTokenMap`
5. **构建QueryResult**：将转账列表构建为 `model.QueryResult` 结构
6. **写入数据库**：调用 `graph.SyncFromQresSimple()` 写入图数据库

## 输出示例

```
输入路径: ./output/17400000_17500000.csv
图数据库路径: /data/mftracer/graph_db
价格数据库路径: /data/mftracer/price_db
找到 1 个CSV文件:
  - ./output/17400000_17500000.csv

正在初始化图数据库...
图数据库初始化成功
正在初始化价格数据库...
价格数据库初始化成功

[1/1] 处理文件: 17400000_17500000.csv
  区块范围: 17400000-17500000, BlockID: 174
  读取到 16454505 条转账记录
  发现 56289 个唯一代币
  已将 56289 个新代币添加到支持列表
  构建 QueryResult 完成
  正在写入图数据库...
  ✅ 处理成功，耗时: 2m30s

============================================================
处理完成！成功: 1, 失败: 0
============================================================
```

## 注意事项

1. **文件命名**：CSV文件必须严格按照 `{startBlock}_{endBlock}.csv` 格式命名
2. **区块范围**：每个CSV文件必须包含恰好100000个区块的数据
3. **代币支持**：工具会自动将所有发现的代币添加到支持列表，避免过滤问题
4. **数据库路径**：确保数据库路径有足够的磁盘空间和写入权限
5. **CSV格式**：CSV文件应使用UTF-8编码，字段之间用逗号分隔

## 错误处理

工具会输出详细的错误信息，常见错误包括：

- **文件名格式错误**：文件名不符合 `{startBlock}_{endBlock}.csv` 格式
- **区块范围错误**：区块范围不是100000个区块
- **CSV解析错误**：CSV行格式不正确或字段数量不匹配
- **数据库错误**：数据库初始化或写入失败

## 技术实现

- **CSV解析**：使用 `codes/data/request.go` 中的 `Res_Transaction_CSVLine_Ver2.Decode()` 方法
- **数据编码**：使用 `codes/encoding` 模块进行数据序列化
- **数据库写入**：使用 `codes/graph/sync.go` 中的 `SyncFromQresSimple()` 函数
- **代币管理**：自动更新 `model.SupportTokenMap` 和 `model.SupportTokenList`

## 相关文档

- `codes/DATA_IMPORT_GUIDE.md` - 数据导入流程详细说明
- `codes/README.md` - 项目结构和模块说明
- `codes/data/request.go` - CSV解析实现
