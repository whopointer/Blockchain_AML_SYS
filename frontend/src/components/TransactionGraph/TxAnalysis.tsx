import React, { useState, useEffect, useMemo } from "react";
import { LinkItem, NodeItem } from "./types";
import { Input, Table, Typography, Space } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import TxDetail from "./TxDetail";

const { Text } = Typography;

interface TxAnalysisProps {
  nodes?: NodeItem[];
  links?: LinkItem[];
}

interface AddressStat {
  address: string;
  count: number;
  totalValue: number;
}

const TxAnalysis: React.FC<TxAnalysisProps> = ({ nodes = [], links = [] }) => {
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

    // 找到对应的链接
    let foundLink: LinkItem | undefined;

    if (isOutgoing) {
      // 查找从当前节点到指定地址的链接
      const targetNode = nodes.find(
        (node) => (node.addr || node.title || node.id) === address
      );

      if (targetNode) {
        foundLink = links.find(
          (link) => link.from === nodes[0]?.id && link.to === targetNode.id
        );
      }
    } else {
      // 查找从指定地址到当前节点的链接
      const sourceNode = nodes.find(
        (node) => (node.addr || node.title || node.id) === address
      );

      if (sourceNode) {
        foundLink = links.find(
          (link) => link.from === sourceNode.id && link.to === nodes[0]?.id
        );
      }
    }

    if (foundLink) {
      setSelectedLink(foundLink);
      setShowTxDetail(true);
    }
  };

  // 计算地址统计数据
  useEffect(() => {
    // 发送地址和接收地址统计: 去除nodes[0]本身
    if (!nodes.length || !links.length) {
      setOutgoingStats([]);
      setIncomingStats([]);
      return;
    }

    const outgoingMap: Record<string, AddressStat> = {};
    const incomingMap: Record<string, AddressStat> = {};

    const currentNodeAddr = nodes[0]?.addr || nodes[0]?.title || nodes[0]?.id;

    links.forEach((link) => {
      // 发送地址统计（来自当前节点的链接）
      if (link.from === nodes[0]?.id) {
        const targetNode = nodes.find((node) => node.id === link.to);
        if (targetNode) {
          const targetAddr =
            targetNode.addr || targetNode.title || targetNode.id;
          // 排除当前节点自身
          if (targetAddr !== currentNodeAddr) {
            if (!outgoingMap[targetAddr]) {
              outgoingMap[targetAddr] = {
                address: targetAddr,
                count: 0,
                totalValue: 0,
              };
            }
            outgoingMap[targetAddr].count += link.tx_hash_list.length;
            outgoingMap[targetAddr].totalValue += link.val;
          }
        }
      }

      // 接收地址统计（发送到当前节点的链接）
      if (link.to === nodes[0]?.id) {
        const sourceNode = nodes.find((node) => node.id === link.from);
        if (sourceNode) {
          const sourceAddr =
            sourceNode.addr || sourceNode.title || sourceNode.id;
          // 排除当前节点自身
          if (sourceAddr !== currentNodeAddr) {
            if (!incomingMap[sourceAddr]) {
              incomingMap[sourceAddr] = {
                address: sourceAddr,
                count: 0,
                totalValue: 0,
              };
            }
            incomingMap[sourceAddr].count += link.tx_hash_list.length;
            incomingMap[sourceAddr].totalValue += link.val;
          }
        }
      }
    });

    const outgoingStatsArray = Object.values(outgoingMap);
    const incomingStatsArray = Object.values(incomingMap);

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
      title: "BNB",
      dataIndex: "totalValue",
      key: "totalValue",
      width: 100,
      sorter: (a: AddressStat, b: AddressStat) => a.totalValue - b.totalValue,
      showSorterTooltip: false,
      render: (value: number) => `${value}`,
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
          onClick={() => handleCountClick(record.address, false)}
          style={{ color: "#667eea", textDecoration: "underline" }}
        >
          {text}
        </a>
      ),
    },
    {
      title: "BNB",
      dataIndex: "totalValue",
      key: "totalValue",
      width: 100,
      sorter: (a: AddressStat, b: AddressStat) => a.totalValue - b.totalValue,
      showSorterTooltip: false,
      render: (value: number) => `${value}`,
    },
  ];

  return (
    <div
      style={{
        backgroundColor: "#244963",
        border: "1px solid #3a5f7f",
        borderRadius: 8,
        padding: 16,
        height: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <p
        style={{
          marginTop: 0,
          marginBottom: 0,
          color: "#ffffff",
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
          color: "#9bb3c8",
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
      />
    </div>
  );
};

export default TxAnalysis;
