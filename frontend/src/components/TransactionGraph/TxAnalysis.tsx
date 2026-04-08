import React, { useState, useEffect, useMemo } from "react";
import { LinkItem, NodeItem } from "../GraphCommon/types";
import { Input, Table, Typography, Space, Card } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import TxDetail from "../GraphCommon/TxDetail";
import { formatEthValue } from "@/utils/ethUtils";

const { Text } = Typography;

interface TxAnalysisProps {
  nodes?: NodeItem[];
  links?: LinkItem[];
  currencySymbol?: string;
}

interface AddressStat {
  address: string;
  count: number;
  totalValue: number;
}

const TxAnalysis: React.FC<TxAnalysisProps> = ({
  nodes = [],
  links = [],
  currencySymbol,
}) => {
  const [searchText, setSearchText] = useState("");
  const [outgoingStats, setOutgoingStats] = useState<AddressStat[]>([]);
  const [incomingStats, setIncomingStats] = useState<AddressStat[]>([]);
  const [showTxDetail, setShowTxDetail] = useState(false);
  const [selectedLink, setSelectedLink] = useState<LinkItem | null>(null);

  // 根据搜索文本过滤统计数据
  const filteredOutgoingStats = useMemo(() => {
    if (!searchText.trim()) {
      return outgoingStats;
    }
    return outgoingStats.filter((stat) => stat.address.includes(searchText));
  }, [outgoingStats, searchText]);

  const filteredIncomingStats = useMemo(() => {
    if (!searchText.trim()) {
      return incomingStats;
    }
    return incomingStats.filter((stat) => stat.address.includes(searchText));
  }, [incomingStats, searchText]);

  // 处理搜索
  const handleSearch = () => {
    // 搜索逻辑已移到useMemo中处理
  };

  // 处理交易数点击
  const handleCountClick = (address: string, isOutgoing: boolean) => {
    if (!nodes.length || !links.length) return;

    const centerNodeId = nodes[0]?.id;
    let foundLinks: LinkItem[] = [];

    if (isOutgoing) {
      // 查找所有从指定地址发出的链接（排除中心节点）
      const sourceNode = nodes.find(
        (node) => (node.addr || node.title || node.id) === address,
      );

      if (sourceNode && sourceNode.id !== centerNodeId) {
        foundLinks = links.filter((link) => link.from === sourceNode.id);
      }
    } else {
      // 查找所有发送到指定地址的链接（排除中心节点）
      const targetNode = nodes.find(
        (node) => (node.addr || node.title || node.id) === address,
      );

      if (targetNode && targetNode.id !== centerNodeId) {
        foundLinks = links.filter((link) => link.to === targetNode.id);
      }
    }

    if (foundLinks.length > 0) {
      // 合并所有交易哈希列表
      const allTxHashes = foundLinks.flatMap((link) => link.tx_hash_list || []);

      // 创建一个包含所有交易哈希的新链接对象
      const mergedLink: LinkItem = {
        ...foundLinks[0],
        tx_hash_list: allTxHashes,
        val: foundLinks.reduce((total, link) => total + (link.val || 0), 0),
      };

      setSelectedLink(mergedLink);
      setShowTxDetail(true);
    }
  };

  // 计算地址统计数据
  useEffect(() => {
    if (!nodes.length || !links.length) {
      setOutgoingStats([]);
      setIncomingStats([]);
      return;
    }

    const outgoingMap: Record<
      string,
      { count: Set<string>; totalValue: number }
    > = {};
    const incomingMap: Record<
      string,
      { count: Set<string>; totalValue: number }
    > = {};

    const centerNodeId = nodes[0]?.id;

    links.forEach((link) => {
      // 发送地址统计：统计所有边的起点节点
      const sourceNode = nodes.find((node) => node.id === link.from);
      if (sourceNode && sourceNode.id !== centerNodeId) {
        const sourceAddr = sourceNode.addr || sourceNode.title || sourceNode.id;
        if (!outgoingMap[sourceAddr]) {
          outgoingMap[sourceAddr] = {
            count: new Set(),
            totalValue: 0,
          };
        }
        link.tx_hash_list.forEach((txHash) =>
          outgoingMap[sourceAddr].count.add(txHash),
        );
        outgoingMap[sourceAddr].totalValue += link.val || 0;
      }

      // 接收地址统计：统计所有边的终点节点
      const targetNode = nodes.find((node) => node.id === link.to);
      if (targetNode && targetNode.id !== centerNodeId) {
        const targetAddr = targetNode.addr || targetNode.title || targetNode.id;
        if (!incomingMap[targetAddr]) {
          incomingMap[targetAddr] = {
            count: new Set(),
            totalValue: 0,
          };
        }
        link.tx_hash_list.forEach((txHash) =>
          incomingMap[targetAddr].count.add(txHash),
        );
        incomingMap[targetAddr].totalValue += link.val || 0;
      }
    });

    const outgoingStatsArray = Object.values(outgoingMap).map((stat, index) => {
      const address = Object.keys(outgoingMap)[index];
      return {
        address: address,
        count: stat.count.size,
        totalValue: stat.totalValue,
        key: address,
      };
    });
    const incomingStatsArray = Object.values(incomingMap).map((stat, index) => {
      const address = Object.keys(incomingMap)[index];
      return {
        address: address,
        count: stat.count.size,
        totalValue: stat.totalValue,
        key: address,
      };
    });

    setOutgoingStats(outgoingStatsArray);
    setIncomingStats(incomingStatsArray);
  }, [nodes, links]);

  // 发送地址表格列定义
  const outgoingColumns = [
    {
      title: "发送地址",
      dataIndex: "address",
      key: "address",
      width: 150,
      render: (text: string) => (
        <Text ellipsis={{ tooltip: text }} style={{ width: 170 }}>
          {text}
        </Text>
      ),
    },
    {
      title: "交易数",
      dataIndex: "count",
      key: "count",
      width: 80,
      sorter: (a: AddressStat, b: AddressStat) => a.count - b.count,
      showSorterTooltip: false,
      render: (text: number, record: AddressStat) => (
        <a
          onClick={() => handleCountClick(record.address, true)}
          style={{ color: "#667eea", textDecoration: "underline" }}
        >
          {text}
        </a>
      ),
    },
    {
      title: currencySymbol || "BTC",
      dataIndex: "totalValue",
      key: "totalValue",
      width: 100,
      sorter: (a: AddressStat, b: AddressStat) => a.totalValue - b.totalValue,
      showSorterTooltip: false,
      render: (value: number) => {
        if (currencySymbol === "ETH") {
          return value.toFixed(6);
        }
        return `${value}`;
      },
    },
  ];

  // 接收地址表格列定义
  const incomingColumns = [
    {
      title: "接收地址",
      dataIndex: "address",
      key: "address",
      width: 150,
      render: (text: string) => (
        <Text ellipsis={{ tooltip: text }} style={{ width: 170 }}>
          {text}
        </Text>
      ),
    },
    {
      title: "交易数",
      dataIndex: "count",
      key: "count",
      width: 80,
      sorter: (a: AddressStat, b: AddressStat) => a.count - b.count,
      showSorterTooltip: false,
      render: (text: number, record: AddressStat) => (
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault();
            handleCountClick(record.address, false);
          }}
        >
          {text}
        </a>
      ),
    },
    {
      title: currencySymbol || "BTC",
      dataIndex: "totalValue",
      key: "totalValue",
      width: 100,
      sorter: (a: AddressStat, b: AddressStat) => a.totalValue - b.totalValue,
      showSorterTooltip: false,
      render: (value: number) => {
        if (currencySymbol === "ETH") {
          return value.toFixed(6);
        }
        return `${value}`;
      },
    },
  ];

  return (
    <Card
      style={{
        backgroundColor: "#ffffff",
        borderRadius: 8,
        height: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <p
        style={{
          marginTop: 0,
          marginBottom: 0,
          color: "#222",
          fontSize: 18,
          fontWeight: 500,
        }}
      >
        交易分析
      </p>
      {/* 当前节点 */}
      <p
        style={{
          marginTop: 0,
          marginBottom: 16,
          fontSize: 12,
          color: "#666",
        }}
      >
        {nodes.length > 0
          ? `${nodes[0].addr || nodes[0].title || nodes[0].id}`
          : ""}
      </p>

      <div style={{ marginBottom: 16 }}>
        <Space direction="vertical" style={{ width: "100%" }}>
          <Input
            placeholder="输入节点地址进行搜索"
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            onPressEnter={handleSearch}
            addonAfter={
              <a onClick={handleSearch} style={{ textDecoration: "none" }}>
                搜索
              </a>
            }
          />
        </Space>
      </div>

      <div style={{ marginBottom: 16 }}>
        <Table
          dataSource={filteredIncomingStats}
          columns={incomingColumns}
          pagination={false}
          size="small"
          scroll={{ y: 200 }}
          style={{ marginTop: 8 }}
        />
      </div>

      <div>
        <Table
          dataSource={filteredOutgoingStats}
          columns={outgoingColumns}
          pagination={false}
          size="small"
          scroll={{ y: 200 }}
          style={{ marginTop: 8 }}
        />
      </div>

      <TxDetail
        show={showTxDetail}
        onHide={() => setShowTxDetail(false)}
        link={selectedLink}
        currencySymbol={currencySymbol}
      />
    </Card>
  );
};

export default TxAnalysis;
