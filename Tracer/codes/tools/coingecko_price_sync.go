package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
	"transfer-graph-evm/model"
	"transfer-graph-evm/pricedb"
	"transfer-graph-evm/utils"
)

type coinListEntry struct {
	ID        string            `json:"id"`
	Symbol    string            `json:"symbol"`
	Name      string            `json:"name"`
	Platforms map[string]string `json:"platforms"`
}

type coinHistoryResponse struct {
	MarketData struct {
		CurrentPrice map[string]float64 `json:"current_price"`
	} `json:"market_data"`
}

func main() {
	var (
		priceDBPath     = flag.String("price_db_path", "", "PriceDB path. If empty, read from config.toml (database.price_db_path)")
		apiKey          = flag.String("api_key", "", "CoinGecko API key (or set COINGECKO_API_KEY)")
		baseURL         = flag.String("base_url", "https://pro-api.coingecko.com/api/v3", "CoinGecko API base URL")
		platform        = flag.String("platform", "ethereum", "Platform key used in /coins/list platforms (e.g. ethereum, polygon-pos)")
		tokenListFile   = flag.String("token_list_file", "", "Token list file: each line '0xTokenAddress' or '0xTokenAddress,anything'")
		blockDatesFile  = flag.String("block_dates_file", "", "Block date mapping file: 'blockID,YYYY-MM-DD' or 'block,timestamp' or 'block,YYYY-MM-DD'")
		startBlock      = flag.Uint64("start_block", 0, "Optional: filter blocks >= start_block (inclusive)")
		endBlock        = flag.Uint64("end_block", 0, "Optional: filter blocks < end_block (exclusive)")
		writeDecimals   = flag.String("decimals_file", "", "Optional decimals file: '0xTokenAddress,decimals' per line")
		rateLimitMs     = flag.Int("rate_limit_ms", 1200, "Min delay between CoinGecko requests (ms)")
		batchSize       = flag.Int("batch_size", 200, "Write batch size for PriceDB")
		dryRun          = flag.Bool("dry_run", false, "If true, only print actions without writing")
	)
	flag.Parse()

	model.InitGlobalTomlConfig()
	utils.SetupLoggers()

	key := strings.TrimSpace(*apiKey)
	if key == "" {
		key = strings.TrimSpace(os.Getenv("COINGECKO_API_KEY"))
	}
	if key == "" {
		fmt.Fprintln(os.Stderr, "❌ missing CoinGecko API key (use -api_key or COINGECKO_API_KEY)")
		os.Exit(1)
	}

	if strings.TrimSpace(*tokenListFile) == "" {
		fmt.Fprintln(os.Stderr, "❌ token_list_file is required")
		os.Exit(1)
	}
	if strings.TrimSpace(*blockDatesFile) == "" {
		fmt.Fprintln(os.Stderr, "❌ block_dates_file is required")
		os.Exit(1)
	}

	p := strings.TrimSpace(*priceDBPath)
	if p == "" {
		p = model.GetConfigPriceDBPath()
	}
	if p == "" {
		fmt.Fprintln(os.Stderr, "❌ price_db_path is empty (set -price_db_path or config.toml)")
		os.Exit(1)
	}

	var pdb *pricedb.PriceDB
	var err error
	if !*dryRun {
		pdb, err = pricedb.NewPriceDB(p, false)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ open PriceDB failed: %v\n", err)
			os.Exit(1)
		}
		defer pdb.Close()
	}

	if strings.TrimSpace(*writeDecimals) != "" {
		if *dryRun {
			fmt.Printf("[dry-run] Would write decimals from %s\n", *writeDecimals)
		} else {
			dir, file := splitDirFile(*writeDecimals)
			if err := pricedb.SimpleSyncDecimals(pdb, dir, file); err != nil {
				fmt.Fprintf(os.Stderr, "❌ sync decimals failed: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("✅ decimals synced from %s\n", *writeDecimals)
		}
	}

	tokens, err := readTokenList(*tokenListFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ read token list failed: %v\n", err)
		os.Exit(1)
	}
	blockIDToDate, err := readBlockDates(*blockDatesFile, *startBlock, *endBlock)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ read block dates failed: %v\n", err)
		os.Exit(1)
	}
	if len(blockIDToDate) == 0 {
		fmt.Fprintln(os.Stderr, "❌ no blockID->date entries found after filtering")
		os.Exit(1)
	}

	idMap, err := fetchCoinList(*baseURL, key, *platform)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ fetch /coins/list failed: %v\n", err)
		os.Exit(1)
	}

	tokenToID := make(map[string]string, len(tokens))
	nativeTokenAddr := strings.ToLower(model.NativeTokenAddress.Hex())
	for _, t := range tokens {
		addr := strings.ToLower(t.Hex())
		// 特殊处理 ETH（原生代币）：CoinGecko ID 固定为 "ethereum"
		if addr == nativeTokenAddr {
			if *platform == "ethereum" {
				tokenToID[addr] = "ethereum"
				continue
			}
			// 如果平台不是 ethereum，跳过 ETH（因为 ETH 只存在于以太坊链）
			fmt.Fprintf(os.Stderr, "⚠️  跳过 ETH（原生代币），因为平台是 %s 而非 ethereum\n", *platform)
			continue
		}
		// 普通 ERC20 代币：通过 platforms 字段匹配
		if id, ok := idMap[addr]; ok {
			tokenToID[addr] = id
		}
	}
	if len(tokenToID) == 0 {
		fmt.Fprintln(os.Stderr, "❌ no token matched CoinGecko IDs. Check platform or token list.")
		os.Exit(1)
	}

	fmt.Printf("PriceDB: %s\n", p)
	fmt.Printf("Platform: %s\n", *platform)
	fmt.Printf("Tokens: %d (matched %d)\n", len(tokens), len(tokenToID))
	fmt.Printf("BlockIDs: %d\n", len(blockIDToDate))

	if *dryRun {
		fmt.Println("[dry-run] Skip fetching and writing prices.")
		return
	}

	client := &http.Client{Timeout: 30 * time.Second}
	ctx := context.Background()
	ticker := time.NewTicker(time.Duration(*rateLimitMs) * time.Millisecond)
	defer ticker.Stop()

	records := make([]*pricedb.WriteRecord, 0, *batchSize)
	flush := func() error {
		if len(records) == 0 {
			return nil
		}
		req := &pricedb.WriteRequest{
			Desc:     "coingecko_price_sync",
			Contents: records,
			Parallel: 1,
		}
		if err := pdb.WriteZ(req, ctx); err != nil {
			return err
		}
		records = records[:0]
		return nil
	}

	for blockID, date := range blockIDToDate {
		dateStr := date.Format("2006-01-02")
		for tokenAddr, coinID := range tokenToID {
			<-ticker.C
			priceUSD, err := fetchCoinHistoryUSD(client, *baseURL, key, coinID, dateStr)
			if err != nil {
				fmt.Fprintf(os.Stderr, "⚠️  price fetch failed (coin=%s date=%s): %v\n", coinID, dateStr, err)
				continue
			}
			priceScaled := priceUSD * float64(pricedb.PriceFactor)
			records = append(records, &pricedb.WriteRecord{
				BlockID: blockID,
				Token:   model.HexToAddress(tokenAddr),
				Price:   priceScaled,
			})
			if len(records) >= *batchSize {
				if err := flush(); err != nil {
					fmt.Fprintf(os.Stderr, "❌ write batch failed: %v\n", err)
					os.Exit(1)
				}
			}
		}
	}
	if err := flush(); err != nil {
		fmt.Fprintf(os.Stderr, "❌ final write failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✅ PriceDB filled from CoinGecko")
}

func fetchCoinList(baseURL, apiKey, platform string) (map[string]string, error) {
	url := strings.TrimRight(baseURL, "/") + "/coins/list?include_platform=true"
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-cg-pro-api-key", apiKey)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode/100 != 2 {
		body, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("coins/list status=%d body=%s", res.StatusCode, string(body))
	}
	var list []coinListEntry
	if err := json.NewDecoder(res.Body).Decode(&list); err != nil {
		return nil, err
	}
	ret := make(map[string]string, len(list))
	for _, entry := range list {
		if entry.Platforms == nil {
			continue
		}
		addr, ok := entry.Platforms[platform]
		if !ok || addr == "" {
			continue
		}
		ret[strings.ToLower(addr)] = entry.ID
	}
	return ret, nil
}

func fetchCoinHistoryUSD(client *http.Client, baseURL, apiKey, coinID, date string) (float64, error) {
	url := fmt.Sprintf("%s/coins/%s/history?date=%s&localization=false", strings.TrimRight(baseURL, "/"), coinID, date)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("x-cg-pro-api-key", apiKey)
	res, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()
	if res.StatusCode/100 != 2 {
		body, _ := io.ReadAll(res.Body)
		return 0, fmt.Errorf("history status=%d body=%s", res.StatusCode, string(body))
	}
	var resp coinHistoryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		return 0, err
	}
	price, ok := resp.MarketData.CurrentPrice["usd"]
	if !ok {
		return 0, fmt.Errorf("usd price missing")
	}
	return price, nil
}

func readTokenList(filePath string) ([]model.Address, error) {
	b, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(b), "\n")
	out := make([]model.Address, 0, len(lines))
	seen := make(map[string]struct{}, len(lines))
	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		addr := ln
		// 支持逗号分隔的注释：0xAddr,comment
		if i := strings.IndexByte(ln, ','); i >= 0 {
			addr = strings.TrimSpace(ln[:i])
		}
		// 支持 # 分隔的注释：0xAddr # comment
		if i := strings.IndexByte(addr, '#'); i >= 0 {
			addr = strings.TrimSpace(addr[:i])
		}
		addr = strings.ToLower(addr)
		if len(addr) != 42 || !strings.HasPrefix(addr, "0x") {
			continue
		}
		if _, ok := seen[addr]; ok {
			continue
		}
		seen[addr] = struct{}{}
		out = append(out, model.HexToAddress(addr))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no valid token address in %s", filePath)
	}
	return out, nil
}

func readBlockDates(filePath string, startBlock, endBlock uint64) (map[uint16]time.Time, error) {
	b, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(b), "\n")
	ret := make(map[uint16]time.Time, len(lines))
	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		parts := strings.Split(ln, ",")
		if len(parts) < 2 {
			continue
		}
		left := strings.TrimSpace(parts[0])
		right := strings.TrimSpace(parts[1])
		if left == "" || right == "" {
			continue
		}
		blockNum, err := strconv.ParseUint(left, 10, 64)
		if err != nil {
			continue
		}
		if startBlock != 0 && blockNum < startBlock {
			continue
		}
		if endBlock != 0 && blockNum >= endBlock {
			continue
		}
		var date time.Time
		if strings.Contains(right, "-") {
			date, err = time.Parse("2006-01-02", right)
			if err != nil {
				continue
			}
		} else {
			ts, err := strconv.ParseInt(right, 10, 64)
			if err != nil {
				continue
			}
			date = time.Unix(ts, 0).UTC()
		}
		blockID := pricedb.GetBlockID(blockNum)
		if _, ok := ret[blockID]; !ok {
			ret[blockID] = date
		}
	}
	return ret, nil
}

func splitDirFile(p string) (dir string, file string) {
	i := strings.LastIndexAny(p, `/\`)
	if i < 0 {
		return ".", p
	}
	if i == 0 {
		return string(p[0]), p[1:]
	}
	return p[:i], p[i+1:]
}

