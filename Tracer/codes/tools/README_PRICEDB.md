## PriceDB（价格数据库）是什么？有什么用？

`pricedb/` 是一个基于 **Pebble** 的本地 KV 数据库，用来存两类信息：

- **Token 价格（USD）**：用于把 `Transfer.Value` / `Tx.Value`（链上整数金额）换算成 **美元价值**，给 `flow/` 模块做阈值过滤（`activate_threshold`）和流量传播计算。
- **Token decimals**：用于把链上整数金额按 decimals 还原到人类可读数量，再换算到美元。

在 `TraceDownstream` 中，价格库在这里被打开并用于构建 `PriceCache`：

```27:39:codes/experiment/trace.go
pDBPath := model.GetConfigPriceDBPath()
p, err := pricedb.NewPriceDB(pDBPath, false)
if err != nil {
    panic(err)
}
defer p.Close()
```

如果 PriceDB 没有对应 token 的价格/decimals，很多边会被视为“无法估值”，进而在 `flow/edges.go` 中被跳过，导致结果可能偏小甚至为 0（尤其当你用美元阈值过滤时）。

---

## PriceDB 的键和值长什么样？

### 1) 价格记录（Price）

- **粒度**：按 `pricedb.BlockSpan = 10000` 对区块号分桶（BlockID = block / 10000）
- **Key**：`'P' + blockID(2 bytes) + keccak(token)[0:8]`
- **Value**：`msgp float64`（价格数值，内部使用 `PriceFactor=1_000_000` 做缩放）

见：`codes/pricedb/model.go` 的 `MakePIDWithBlockID`/`MakePID`。

### 2) decimals 记录（Decimal）

- **Key**：`'D' + token(20 bytes)`
- **Value**：`msgp byte`（uint8 decimals）

见：`codes/pricedb/model.go` 的 `MakeDID`，以及 `codes/pricedb/db.go` 的 `SimpleWriteDecimals`/`SimpleReadAllDecimals`。

---

## 如何填充 PriceDB（推荐流程）

PriceDB 的填充分两步：**先 decimals，再 prices**。

### 步骤 0：配置路径

编辑 `codes/config.toml`：

```toml
[database]
price_db_path = "D:\\MFTracer\\price_db"   # Windows 示例
# price_db_path = "/data/mftracer/price_db" # Linux 示例
```

### 步骤 1：写入 decimals（从文件导入）

准备一个 decimals 文件（每行一条）：

```text
0xdAC17F958D2ee523a2206206994597C13D831ec7,6
0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48,6
0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2,18
```

然后运行工具 `price_sync`（见下一节）。

### 步骤 2：写入 prices（按区块范围批量同步）

项目内置的 price 拉取逻辑在 `codes/pricedb/request.go`，它会请求本地服务：

- `priceServiceUrl = "http://localhost:7001/api/tokenpricebulkV2"`

同时 `pricedb.SyncByOpenSearch(...)` 需要 OpenSearch（用于在每个 10k 桶里找一个“有交易的区块”，拿到 timestamp）。

---

## 使用工具 `codes/tools/price_sync.go`

编译：

```bash
cd codes/tools
go build -o price_sync price_sync.go
```

### 仅写 decimals（不写价格）

```bash
./price_sync -decimals_file=./decimals.csv
```

### 写 decimals + 写价格（需要 OpenSearch + 本地 price service）

准备 token 列表文件（每行一个 token 地址即可）：

```text
0xdAC17F958D2ee523a2206206994597C13D831ec7
0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
```

运行（以 ETH 为例）：

```bash
./price_sync \
  -decimals_file=./decimals.csv \
  -token_list_file=./tokens.txt \
  -start_block=17300000 \
  -end_block=17600000 \
  -opensearch_index=eth_block \
  -opensearch_url=http://127.0.0.1:9200 \
  -opensearch_user=... \
  -opensearch_password=...
```

如果你只想先看会做什么：

```bash
./price_sync -dry_run -decimals_file=./decimals.csv -token_list_file=./tokens.txt -start_block=17300000 -end_block=17600000 -opensearch_index=eth_block -opensearch_url=http://127.0.0.1:9200
```

---

## 使用工具 `codes/tools/coingecko_price_sync.go`（CoinGecko API）

此工具基于 CoinGecko 的两个 API：

- `/coins/list?include_platform=true`：把合约地址映射到 CoinGecko 的 coin id
- `/coins/{id}/history?date=YYYY-MM-DD`：获取某日期的 USD 价格

> 注意：`history` 的价格是 **当天 00:00 UTC 的快照**。因此你需要提供 “某个 block 对应的日期”。

### 1) 准备 token 列表文件

每行一个代币地址（支持注释，以 `#` 开头）：

```text
# USDT
0xdAC17F958D2ee523a2206206994597C13D831ec7
# WETH
0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
# ETH（原生代币，使用全零地址）
0x0000000000000000000000000000000000000000
```

**格式要求：**
- 每行一个以太坊地址（42 字符，以 `0x` 开头）
- 支持在地址后添加逗号和注释（会被忽略）
- 空行和以 `#` 开头的行会被跳过
- 地址会自动转换为小写并去重

**ETH（原生代币）的特殊处理：**
- ETH 在代码中用全零地址 `0x0000000000000000000000000000000000000000` 表示
- 工具会自动识别 ETH 并使用 CoinGecko ID `"ethereum"` 获取价格
- **注意**：只有当 `-platform=ethereum` 时，ETH 才会被处理（因为 ETH 只存在于以太坊链）

### 2) 准备 block→date 映射文件

支持以下任一格式（CSV，每行一条）：

- `block,YYYY-MM-DD`
- `block,timestamp`（Unix 秒）
- `blockID,YYYY-MM-DD`（blockID=block/10000）

示例：

```text
17300000,2023-05-20
17300001,1684540800
17310000,2023-05-21
```

### 3) 运行

```bash
cd codes/tools
go build -o coingecko_price_sync coingecko_price_sync.go

export COINGECKO_API_KEY=your_key
./coingecko_price_sync \
  -token_list_file=./tokens.txt \
  -block_dates_file=./block_dates.csv \
  -platform=ethereum \
  -rate_limit_ms=1200
```

可选项：

- `-decimals_file=...`：顺便写 decimals
- `-price_db_path=...`：不从 `config.toml` 读取
- `-start_block / -end_block`：对 block 映射做区间过滤
- `-base_url=...`：默认用 Pro API

## 验证 PriceDB 是否写入成功（建议）

最简单的验证方式是：重新跑一次 `mftracer`，观察输出 CSV 不再全为 0，且日志不再出现大量“price missing / decimals missing”导致的过滤（如果你加了 debug 日志）。

