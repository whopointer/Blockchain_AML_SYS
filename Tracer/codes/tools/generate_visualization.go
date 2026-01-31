package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type NodeInfo struct {
	Address   string
	TotalIn   float64
	TotalOut  float64
	Remaining float64
	IsSource  bool
}

type EdgeInfo struct {
	From      string
	To        string
	Value     float64
	UsedValue float64
	Timestamp string
	TxHash    string
	Age       int
}

func main() {
	var (
		alleFile    = flag.String("alle", "./results/trace_result.alle.csv", "所有边的CSV文件路径")
		allnFile    = flag.String("alln", "./results/trace_result.alln.csv", "所有节点的CSV文件路径")
		outputDot   = flag.String("output", "./results/flow_graph.dot", "输出的DOT文件路径")
		maxNodes    = flag.Int("max_nodes", 500, "最大节点数量（超过此数量将只显示重要节点）")
		minValue    = flag.Float64("min_value", 100.0, "最小交易金额（USD），低于此金额的边将被过滤")
		layout      = flag.String("layout", "fdp", "Graphviz布局引擎：dot, neato, fdp, sfdp, twopi, circo")
		showLabels  = flag.Bool("show_labels", true, "是否在边上显示金额标签")
		simplify    = flag.Bool("simplify", false, "是否简化图（合并小金额边）")
	)
	flag.Parse()

	fmt.Printf("========================================\n")
	fmt.Printf("Graphviz 资金流可视化生成工具\n")
	fmt.Printf("========================================\n")
	fmt.Printf("边文件: %s\n", *alleFile)
	fmt.Printf("节点文件: %s\n", *allnFile)
	fmt.Printf("输出文件: %s\n", *outputDot)
	fmt.Printf("最大节点数: %d\n", *maxNodes)
	fmt.Printf("最小金额: %.2f USD\n", *minValue)
	fmt.Printf("布局引擎: %s\n", *layout)
	fmt.Printf("========================================\n\n")

	// 读取节点信息
	fmt.Printf("[1/4] 读取节点信息...\n")
	nodes, err := readNodeFile(*allnFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 读取节点文件失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✅ 读取了 %d 个节点\n\n", len(nodes))

	// 读取边信息
	fmt.Printf("[2/4] 读取边信息...\n")
	edges, err := readEdgeFile(*alleFile, *minValue)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 读取边文件失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✅ 读取了 %d 条边（过滤后）\n\n", len(edges))

	// 构建图结构并过滤
	fmt.Printf("[3/4] 构建图结构...\n")
	nodeSet, edgeSet := buildGraph(nodes, edges, *maxNodes)
	fmt.Printf("✅ 选择了 %d 个节点和 %d 条边用于可视化\n\n", len(nodeSet), len(edgeSet))

	// 生成DOT文件
	fmt.Printf("[4/4] 生成Graphviz DOT文件...\n")
	err = generateDOTFile(*outputDot, nodeSet, edgeSet, nodes, *showLabels, *simplify, *layout)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ 生成DOT文件失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("✅ DOT文件已生成: %s\n\n", *outputDot)

	fmt.Printf("========================================\n")
	fmt.Printf("✅ 生成完成！\n")
	fmt.Printf("========================================\n")
	fmt.Printf("下一步：使用Graphviz转换为SVG\n")
	fmt.Printf("命令：\n")
	fmt.Printf("  %s -Tsvg %s -o %s\n", *layout, *outputDot, strings.Replace(*outputDot, ".dot", ".svg", 1))
	fmt.Printf("\n或者转换为PNG：\n")
	fmt.Printf("  %s -Tpng %s -o %s\n", *layout, *outputDot, strings.Replace(*outputDot, ".dot", ".png", 1))
	fmt.Printf("========================================\n")
}

func readNodeFile(filePath string) (map[string]*NodeInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	nodes := make(map[string]*NodeInfo)
	for i, record := range records {
		if i == 0 {
			continue // 跳过标题行
		}
		if len(record) < 4 {
			continue
		}

		address := strings.TrimSpace(record[0])
		totalIn, _ := strconv.ParseFloat(record[1], 64)
		totalOut, _ := strconv.ParseFloat(record[2], 64)
		remaining, _ := strconv.ParseFloat(record[3], 64)

		isSource := totalIn == 0.0 && totalOut > 0.0

		nodes[address] = &NodeInfo{
			Address:   address,
			TotalIn:   totalIn,
			TotalOut:  totalOut,
			Remaining: remaining,
			IsSource:  isSource,
		}
	}

	return nodes, nil
}

func readEdgeFile(filePath string, minValue float64) ([]*EdgeInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	edges := make([]*EdgeInfo, 0)
	for i, record := range records {
		if i == 0 {
			continue // 跳过标题行
		}
		if len(record) < 8 {
			continue
		}

		from := strings.TrimSpace(record[1])
		to := strings.TrimSpace(record[2])
		value, _ := strconv.ParseFloat(record[3], 64)
		usedValue, _ := strconv.ParseFloat(record[4], 64)
		timestamp := strings.TrimSpace(record[5])
		txHash := strings.TrimSpace(record[6])
		age, _ := strconv.Atoi(record[7])

		// 过滤小金额边
		if usedValue < minValue {
			continue
		}

		edges = append(edges, &EdgeInfo{
			From:      from,
			To:        to,
			Value:     value,
			UsedValue: usedValue,
			Timestamp: timestamp,
			TxHash:    txHash,
			Age:       age,
		})
	}

	return edges, nil
}

func buildGraph(nodes map[string]*NodeInfo, edges []*EdgeInfo, maxNodes int) (map[string]bool, []*EdgeInfo) {
	// 统计节点重要性（按流入金额和剩余金额）
	nodeImportance := make(map[string]float64)
	for addr, node := range nodes {
		// 重要性 = total_in + abs(remaining) * 2（剩余金额更重要）
		nodeImportance[addr] = node.TotalIn + math.Abs(node.Remaining)*2
	}

	// 统计边的目标节点
	edgeTargets := make(map[string]float64)
	for _, edge := range edges {
		edgeTargets[edge.To] += edge.UsedValue
	}

	// 选择重要节点
	selectedNodes := make(map[string]bool)
	
	// 1. 优先选择源地址
	for addr, node := range nodes {
		if node.IsSource {
			selectedNodes[addr] = true
		}
	}

	// 2. 选择重要节点（按重要性排序）
	type nodeScore struct {
		addr      string
		importance float64
	}
	scores := make([]nodeScore, 0, len(nodeImportance))
	for addr, importance := range nodeImportance {
		scores = append(scores, nodeScore{addr, importance})
	}
	
	// 按重要性排序
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[i].importance < scores[j].importance {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	// 选择前N个节点
	for _, score := range scores {
		if len(selectedNodes) >= maxNodes {
			break
		}
		selectedNodes[score.addr] = true
	}

	// 3. 确保边的两端节点都被包含
	selectedEdges := make([]*EdgeInfo, 0)
	for _, edge := range edges {
		if selectedNodes[edge.From] || selectedNodes[edge.To] {
			// 如果边的端点不在选中列表中，添加它们
			if !selectedNodes[edge.From] {
				selectedNodes[edge.From] = true
			}
			if !selectedNodes[edge.To] {
				selectedNodes[edge.To] = true
			}
			selectedEdges = append(selectedEdges, edge)
		}
	}

	return selectedNodes, selectedEdges
}

func generateDOTFile(outputPath string, nodeSet map[string]bool, edges []*EdgeInfo, nodes map[string]*NodeInfo, showLabels, simplify bool, layout string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// 计算节点大小和颜色的范围
	maxTotalIn := 0.0
	maxRemaining := 0.0
	for addr := range nodeSet {
		if node, ok := nodes[addr]; ok {
			if node.TotalIn > maxTotalIn {
				maxTotalIn = node.TotalIn
			}
			if math.Abs(node.Remaining) > maxRemaining {
				maxRemaining = math.Abs(node.Remaining)
			}
		}
	}

	// 计算边的最大金额
	maxEdgeValue := 0.0
	for _, edge := range edges {
		if edge.UsedValue > maxEdgeValue {
			maxEdgeValue = edge.UsedValue
		}
	}

	// 写入DOT文件头
	fmt.Fprintf(writer, "digraph FlowGraph {\n")
	fmt.Fprintf(writer, "  rankdir=LR;\n")
	fmt.Fprintf(writer, "  node [shape=box, style=rounded];\n")
	fmt.Fprintf(writer, "  edge [fontsize=10];\n\n")

	// 写入节点
	fmt.Fprintf(writer, "  // 节点定义\n")
	for addr := range nodeSet {
		node, ok := nodes[addr]
		if !ok {
			continue
		}

		// 节点大小（基于total_in）
		nodeSize := 1.0
		if maxTotalIn > 0 {
			nodeSize = 0.5 + (node.TotalIn/maxTotalIn)*1.5
		}

		// 节点颜色
		var fillColor string
		if node.IsSource {
			fillColor = "#f0eae1" // 源地址：浅棕色
		} else if node.Remaining > 1000 {
			fillColor = "#ffcccc" // 资金沉淀地址：浅红色
		} else if node.TotalIn > 0 && node.TotalOut > 0 {
			fillColor = "#ccffcc" // 中转地址：浅绿色
		} else {
			fillColor = "#f5f5f5" // 普通地址：浅灰色
		}

		// 节点标签（缩短地址）
		label := shortenAddress(addr)
		if node.IsSource {
			label = "SRC: " + label
		}

		// 节点工具提示
		tooltip := fmt.Sprintf("地址: %s\\n流入: %.2f USD\\n流出: %.2f USD\\n剩余: %.2f USD", 
			addr, node.TotalIn, node.TotalOut, node.Remaining)

		fmt.Fprintf(writer, "  \"%s\" [\n", escapeString(addr))
		fmt.Fprintf(writer, "    label=\"%s\";\n", escapeString(label))
		fmt.Fprintf(writer, "    width=%.2f;\n", nodeSize)
		fmt.Fprintf(writer, "    height=%.2f;\n", nodeSize*0.6)
		fmt.Fprintf(writer, "    fillcolor=\"%s\";\n", fillColor)
		fmt.Fprintf(writer, "    style=\"filled,rounded\";\n")
		fmt.Fprintf(writer, "    tooltip=\"%s\";\n", escapeString(tooltip))
		fmt.Fprintf(writer, "    URL=\"https://etherscan.io/address/%s\";\n", addr)
		fmt.Fprintf(writer, "  ];\n\n")
	}

	// 写入边
	fmt.Fprintf(writer, "  // 边定义\n")
	
	// 如果简化模式，合并相同方向的边
	edgeMap := make(map[string]*EdgeInfo)
	if simplify {
		for _, edge := range edges {
			key := edge.From + "->" + edge.To
			if existing, ok := edgeMap[key]; ok {
				existing.UsedValue += edge.UsedValue
			} else {
				edgeMap[key] = &EdgeInfo{
					From:      edge.From,
					To:        edge.To,
					UsedValue: edge.UsedValue,
					Age:       edge.Age,
				}
			}
		}
		edges = make([]*EdgeInfo, 0, len(edgeMap))
		for _, edge := range edgeMap {
			edges = append(edges, edge)
		}
	}

	for _, edge := range edges {
		if !nodeSet[edge.From] || !nodeSet[edge.To] {
			continue
		}

		// 边宽度（基于金额）
		edgeWidth := 1.0
		if maxEdgeValue > 0 {
			edgeWidth = 0.5 + (edge.UsedValue/maxEdgeValue)*3.0
		}

		// 边颜色（基于跳数）
		var edgeColor string
		switch edge.Age {
		case 1:
			edgeColor = "#5873e0" // 第一跳：蓝色
		case 2:
			edgeColor = "#8b73e0" // 第二跳：紫色
		case 3:
			edgeColor = "#b873e0" // 第三跳：紫红色
		default:
			edgeColor = "#d0d0d0" // 其他：灰色
		}

		// 边标签
		label := ""
		if showLabels {
			if edge.UsedValue >= 1000 {
				label = fmt.Sprintf("%.0fK", edge.UsedValue/1000)
			} else {
				label = fmt.Sprintf("%.0f", edge.UsedValue)
			}
		}

		fmt.Fprintf(writer, "  \"%s\" -> \"%s\" [\n", escapeString(edge.From), escapeString(edge.To))
		fmt.Fprintf(writer, "    penwidth=%.2f;\n", edgeWidth)
		fmt.Fprintf(writer, "    color=\"%s\";\n", edgeColor)
		if label != "" {
			fmt.Fprintf(writer, "    label=\"%s\";\n", escapeString(label))
		}
		fmt.Fprintf(writer, "    tooltip=\"金额: %.2f USD\\n跳数: %d\\n交易: %s\";\n", 
			edge.UsedValue, edge.Age, edge.TxHash)
		fmt.Fprintf(writer, "    URL=\"https://etherscan.io/tx/%s\";\n", edge.TxHash)
		fmt.Fprintf(writer, "  ];\n\n")
	}

	fmt.Fprintf(writer, "}\n")
	return nil
}

func shortenAddress(addr string) string {
	if len(addr) < 10 {
		return addr
	}
	return addr[:6] + "..." + addr[len(addr)-4:]
}

func escapeString(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	return s
}
