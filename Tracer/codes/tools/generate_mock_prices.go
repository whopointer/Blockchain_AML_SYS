package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

/**
 * Generate a mock prices_export.csv file for testing PriceDB import.
 * 
 * This tool reads:
 * - toekn.txt: token addresses
 * - block_dates_10k.csv: block number (already aligned to 10000) to date mapping
 * 
 * Output: prices_export.csv with virtual prices
 */
func main() {
	// 读取代币列表
	tokens, err := readTokenList("./toekn.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error reading toekn.txt: %v\n", err)
		os.Exit(1)
	}
	if len(tokens) == 0 {
		fmt.Fprintf(os.Stderr, "❌ No tokens found in toekn.txt\n")
		os.Exit(1)
	}

	// 虚拟价格（USD）- 基于 2023年5-6月的典型价格
	virtualPrices := map[string]float64{
		"0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 1.0,   // USDC
		"0xdac17f958d2ee523a2206206994597c13d831ec7": 1.0,   // USDT
		"0x6b175474e89094c44da98b954eedeac495271d0f": 1.0,   // DAI
		"0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 2000.0, // WETH (≈ETH price)
		"0x0000000000000000000000000000000000000000": 2000.0, // ETH
	}

	// 读取 block_dates_10k.csv 并提取 blockID->date 映射
	// 注意：block_dates_10k.csv 中的 blockNumber 已经是按 10000 对齐的（如 17300000, 17310000, ...）
	blockIDToDate, err := readBlockDates("./block_dates_10k.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error reading block_dates_10k.csv: %v\n", err)
		os.Exit(1)
	}
	if len(blockIDToDate) == 0 {
		fmt.Fprintf(os.Stderr, "❌ No blockID->date entries found\n")
		os.Exit(1)
	}

	// 生成 prices_export.csv
	out, err := os.Create("./prices_export.csv")
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Error creating prices_export.csv: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	writer := bufio.NewWriter(out)
	writer.WriteString("block_id,token_address,coin_id,date,price_usd\n")

	totalRecords := 0
	for blockID, date := range blockIDToDate {
		for _, token := range tokens {
			price, ok := virtualPrices[token]
			if !ok {
				// 如果代币不在价格映射中，使用默认价格 1.0
				price = 1.0
			}
			coinID := token // 虚拟数据中 coin_id 使用 token 地址
			writer.WriteString(fmt.Sprintf("%d,%s,%s,%s,%.8f\n",
				blockID, token, coinID, date, price))
			totalRecords++
		}
	}

	writer.Flush()
	fmt.Printf("✅ Generated prices_export.csv\n")
	fmt.Printf("   BlockIDs: %d unique blocks\n", len(blockIDToDate))
	fmt.Printf("   Tokens: %d\n", len(tokens))
	fmt.Printf("   Total records: %d\n", totalRecords)
}

func readTokenList(filename string) ([]string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	tokens := []string{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// 支持逗号和 # 注释
		if i := strings.IndexByte(line, ','); i >= 0 {
			line = strings.TrimSpace(line[:i])
		}
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = strings.TrimSpace(line[:i])
		}
		line = strings.ToLower(line)
		if len(line) == 42 && strings.HasPrefix(line, "0x") {
			tokens = append(tokens, line)
		}
	}
	return tokens, scanner.Err()
}

func readBlockDates(filename string) (map[int]string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	blockIDToDate := make(map[int]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		blockNum, err := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 64)
		if err != nil {
			continue
		}
		// 计算 blockID = blockNumber / 10000
		blockID := int(blockNum / 10000)
		date := strings.TrimSpace(parts[1])
		// 只保留第一次出现的日期（按桶起始块）
		if _, ok := blockIDToDate[blockID]; !ok {
			blockIDToDate[blockID] = date
		}
	}
	return blockIDToDate, scanner.Err()
}
