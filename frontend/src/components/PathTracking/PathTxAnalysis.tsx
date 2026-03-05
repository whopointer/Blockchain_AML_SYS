import React, { useState, useEffect, useMemo } from "react";
import { LinkItem, NodeItem } from "../GraphCommon/types";
import { Input, Table, Typography, Space, Card } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import TxDetail from "../GraphCommon/TxDetail";

const { Text } = Typography;

interface PathTxAnalysisProps {
  nodes?: NodeItem[];
  links?: LinkItem[];
  currencySymbol?: string;
}

interface AddressStat {
  address: string;
  count: number;
  totalValue: number;
}

const PathTxAnalysis: React.FC<PathTxAnalysisProps> = ({
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

    // 找到对应的链接
    let foundLink: LinkItem | undefined;

    if (isOutgoing) {
      // 查找从指定地址发出的链接
      const sourceNode = nodes.find(
        (node) => (node.addr || node.title || node.id) === address,
      );

      if (sourceNode) {
        foundLink = links.find((link) => link.from === sourceNode.id);
      }
    } else {
      // 查找发送到指定地址的链接
      const targetNode = nodes.find(
        (node) => (node.addr || node.title || node.id) === address,
      );

      if (targetNode) {
        foundLink = links.find((link) => link.to === targetNode.id);
      }
    }

    if (foundLink) {
      setSelectedLink(foundLink);
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

    const outgoingMap: Record<string, AddressStat> = {};
    const incomingMap: Record<string, AddressStat> = {};

    links.forEach((link) => {
      // 发送地址统计
      const sourceNode = nodes.find((node) => node.id === link.from);
      if (sourceNode) {
        const sourceAddr = sourceNode.addr || sourceNode.title || sourceNode.id;
        if (!outgoingMap[sourceAddr]) {
          outgoingMap[sourceAddr] = {
            address: sourceAddr,
            count: 0,
            totalValue: 0,
          };
        }
        outgoingMap[sourceAddr].count += link.tx_hash_list.length;
        outgoingMap[sourceAddr].totalValue += link.val || 0;
      }

      // 接收地址统计
      const targetNode = nodes.find((node) => node.id === link.to);
      if (targetNode) {
        const targetAddr = targetNode.addr || targetNode.title || targetNode.id;
        if (!incomingMap[targetAddr]) {
          incomingMap[targetAddr] = {
            address: targetAddr,
            count: 0,
            totalValue: 0,
          };
        }
        incomingMap[targetAddr].count += link.tx_hash_list.length;
        incomingMap[targetAddr].totalValue += link.val || 0;
      }
    });

    const outgoingStatsArray = Object.values(outgoingMap).map((stat) => ({
      ...stat,
      key: stat.address,
    }));
    const incomingStatsArray = Object.values(incomingMap).map((stat) => ({
      ...stat,
      key: stat.address,
    }));

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
        路径交易分析
      </p>
      <p
        style={{
          marginTop: 0,
          marginBottom: 16,
          fontSize: 12,
          color: "#666",
        }}
      >
        分析所有地址的交易情况
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

export default PathTxAnalysis;
