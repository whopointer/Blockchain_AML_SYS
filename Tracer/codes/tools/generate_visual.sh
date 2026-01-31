#!/bin/bash

# Graphviz 资金流可视化一键生成脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
ALLE_FILE="../results/trace_result.alle.csv"
ALLN_FILE="../results/trace_result.alln.csv"
OUTPUT_DOT="../results/flow_graph.dot"
OUTPUT_SVG="../results/flow_graph.svg"
MAX_NODES=500
MIN_VALUE=100.0
LAYOUT="fdp"
SHOW_LABELS=true
SIMPLIFY=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -alle)
            ALLE_FILE="$2"
            shift 2
            ;;
        -alln)
            ALLN_FILE="$2"
            shift 2
            ;;
        -output)
            OUTPUT_DOT="$2"
            OUTPUT_SVG="${2%.dot}.svg"
            shift 2
            ;;
        -max_nodes)
            MAX_NODES="$2"
            shift 2
            ;;
        -min_value)
            MIN_VALUE="$2"
            shift 2
            ;;
        -layout)
            LAYOUT="$2"
            shift 2
            ;;
        -simplify)
            SIMPLIFY=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -alle FILE         边CSV文件路径 (默认: $ALLE_FILE)"
            echo "  -alln FILE         节点CSV文件路径 (默认: $ALLN_FILE)"
            echo "  -output FILE       输出DOT文件路径 (默认: $OUTPUT_DOT)"
            echo "  -max_nodes N       最大节点数 (默认: $MAX_NODES)"
            echo "  -min_value N       最小金额阈值 (默认: $MIN_VALUE)"
            echo "  -layout ENGINE     布局引擎: fdp, dot, neato, sfdp (默认: $LAYOUT)"
            echo "  -simplify          简化图（合并重复边）"
            echo "  -h, --help         显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Graphviz 资金流可视化生成"
echo "========================================"
echo ""

# 检查文件是否存在
if [ ! -f "$ALLE_FILE" ]; then
    echo -e "${RED}错误: 边文件不存在: $ALLE_FILE${NC}"
    exit 1
fi

if [ ! -f "$ALLN_FILE" ]; then
    echo -e "${RED}错误: 节点文件不存在: $ALLN_FILE${NC}"
    exit 1
fi

# 检查Graphviz是否安装
if ! command -v $LAYOUT &> /dev/null; then
    echo -e "${RED}错误: Graphviz未安装或 $LAYOUT 命令不可用${NC}"
    echo "请安装Graphviz:"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  macOS: brew install graphviz"
    echo "  Windows: choco install graphviz"
    exit 1
fi

# 检查Go工具是否编译
if [ ! -f "./generate_visualization" ]; then
    echo -e "${YELLOW}警告: generate_visualization 未编译，正在编译...${NC}"
    go build -o generate_visualization generate_visualization.go
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 编译失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}编译成功${NC}"
    echo ""
fi

# 生成DOT文件
echo -e "${GREEN}[1/2] 生成DOT文件...${NC}"
SIMPLIFY_FLAG=""
if [ "$SIMPLIFY" = true ]; then
    SIMPLIFY_FLAG="-simplify"
fi

SHOW_LABELS_FLAG=""
if [ "$SHOW_LABELS" = true ]; then
    SHOW_LABELS_FLAG="-show_labels"
fi

./generate_visualization \
    -alle="$ALLE_FILE" \
    -alln="$ALLN_FILE" \
    -output="$OUTPUT_DOT" \
    -max_nodes=$MAX_NODES \
    -min_value=$MIN_VALUE \
    -layout="$LAYOUT" \
    $SHOW_LABELS_FLAG \
    $SIMPLIFY_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: DOT文件生成失败${NC}"
    exit 1
fi

echo ""

# 转换为SVG
echo -e "${GREEN}[2/2] 转换为SVG...${NC}"
$LAYOUT -Tsvg "$OUTPUT_DOT" -o "$OUTPUT_SVG"

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: SVG转换失败${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}✅ 生成完成！${NC}"
echo "========================================"
echo "DOT文件: $OUTPUT_DOT"
echo "SVG文件: $OUTPUT_SVG"
echo ""
echo "在浏览器中打开SVG文件查看结果"
echo "========================================"
