package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
	"transfer-graph-evm/model"
	"transfer-graph-evm/pricedb"
	"transfer-graph-evm/utils"
)

// Step 2 (offline machine):
// Import exported CoinGecko prices CSV into local PriceDB (Pebble).
//
// Input CSV (header optional):
//   block_id,token_address,coin_id,date,price_usd
// Or at minimum:
//   block_id,token_address,price_usd
//
// Notes:
// - PriceDB stores (blockID, token)->priceScaled where priceScaled = priceUSD * PriceFactor
// - This importer accepts "blockID" OR a real "blockNumber" in the first column:
//   if value > 1_000_000 it will be treated as blockNumber and converted to blockID by /10000.
func main() {
	var (
		priceDBPath   = flag.String("price_db_path", "", "PriceDB path. If empty, read from config.toml (database.price_db_path)")
		inputCSV      = flag.String("input_csv", "", "Input prices CSV file (exported by CoinGeckoPriceExport.java)")
		decimalsFile  = flag.String("decimals_file", "", "Optional decimals file: '0xTokenAddress,decimals' per line (same as pricedb.SimpleSyncDecimals)")
		batchSize     = flag.Int("batch_size", 2000, "Write batch size")
		skipHeader    = flag.Bool("skip_header", true, "Skip the first line as header (safe even if no header)")
		dryRun        = flag.Bool("dry_run", false, "Dry run: parse and print stats, no DB write")
	)
	flag.Parse()

	model.InitGlobalTomlConfig()
	utils.SetupLoggers()

	if strings.TrimSpace(*inputCSV) == "" {
		fmt.Fprintln(os.Stderr, "❌ input_csv is required")
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

	ctx := context.Background()

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

	if strings.TrimSpace(*decimalsFile) != "" {
		if *dryRun {
			fmt.Printf("[dry-run] Would sync decimals from %s\n", *decimalsFile)
		} else {
			dir, file := splitDirFile(*decimalsFile)
			if err := pricedb.SimpleSyncDecimals(pdb, dir, file); err != nil {
				fmt.Fprintf(os.Stderr, "❌ sync decimals failed: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("✅ decimals synced from %s\n", *decimalsFile)
		}
	}

	f, err := os.Open(*inputCSV)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ open input_csv failed: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.ReuseRecord = true

	total := 0
	ok := 0
	skipped := 0
	bad := 0

	records := make([]*pricedb.WriteRecord, 0, *batchSize)
	flush := func() error {
		if *dryRun || len(records) == 0 {
			records = records[:0]
			return nil
		}
		req := &pricedb.WriteRequest{
			Desc:     "pricedb_import_prices",
			Contents: records,
			Parallel: 1,
		}
		if err := pdb.WriteZ(req, ctx); err != nil {
			return err
		}
		records = make([]*pricedb.WriteRecord, 0, *batchSize)
		return nil
	}

	lineNum := 0
	start := time.Now()
	for {
		rec, err := r.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			fmt.Fprintf(os.Stderr, "⚠️  csv read error at line %d: %v\n", lineNum+1, err)
			bad++
			break
		}
		lineNum++
		total++

		if lineNum == 1 && *skipHeader {
			// if first column isn't numeric, treat as header
			if len(rec) > 0 {
				if _, err := strconv.ParseUint(strings.TrimSpace(rec[0]), 10, 64); err != nil {
					skipped++
					continue
				}
			}
		}

		if len(rec) < 3 {
			skipped++
			continue
		}

		blockRaw, err := strconv.ParseUint(strings.TrimSpace(rec[0]), 10, 64)
		if err != nil {
			skipped++
			continue
		}
		var blockID uint16
		if blockRaw > 1_000_000 {
			blockID = pricedb.GetBlockID(blockRaw)
		} else {
			if blockRaw > 0xFFFF {
				skipped++
				continue
			}
			blockID = uint16(blockRaw)
		}

		tokenStr := strings.ToLower(strings.TrimSpace(rec[1]))
		if tokenStr == "" {
			skipped++
			continue
		}
		// accept token like "0x.." only
		if len(tokenStr) != 42 || !strings.HasPrefix(tokenStr, "0x") {
			skipped++
			continue
		}
		token := model.HexToAddress(tokenStr)

		// price_usd may be in col 2 or later depending on export columns
		priceStr := strings.TrimSpace(rec[2])
		if len(rec) >= 5 {
			// Java export format: block_id,token,coin_id,date,price_usd
			priceStr = strings.TrimSpace(rec[4])
		}

		priceUSD, err := strconv.ParseFloat(priceStr, 64)
		if err != nil || priceUSD <= 0 {
			skipped++
			continue
		}
		priceScaled := priceUSD * float64(pricedb.PriceFactor)

		if *dryRun {
			ok++
			continue
		}
		records = append(records, &pricedb.WriteRecord{
			BlockID: blockID,
			Token:   token,
			Price:   priceScaled,
		})
		ok++
		if len(records) >= *batchSize {
			if err := flush(); err != nil {
				fmt.Fprintf(os.Stderr, "❌ write batch failed: %v\n", err)
				os.Exit(1)
			}
		}
	}
	if err := flush(); err != nil {
		fmt.Fprintf(os.Stderr, "❌ final write failed: %v\n", err)
		os.Exit(1)
	}

	dur := time.Since(start)
	if *dryRun {
		fmt.Printf("✅ dry-run parsed. total=%d ok=%d skipped=%d bad=%d duration=%s\n", total, ok, skipped, bad, dur)
	} else {
		fmt.Printf("✅ import done. PriceDB=%s total=%d ok=%d skipped=%d bad=%d duration=%s\n", p, total, ok, skipped, bad, dur)
	}
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

