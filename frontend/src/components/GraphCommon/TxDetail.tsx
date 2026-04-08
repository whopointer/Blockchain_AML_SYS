import React, { useState, useEffect, useMemo } from "react";
import { Modal } from "react-bootstrap";
import {
  Table,
  Tag,
  TableProps,
  Space,
  Tooltip,
  message,
  ConfigProvider,
  Spin,
} from "antd";
import { CopyOutlined } from "@ant-design/icons";
import { LinkItem } from "./types";
import { transactionApi } from "@/services/transaction";

// 自定义函数用于在文本中间添加省略号
const truncateMiddle = (str: string, maxLength: number = 15) => {
  if (str.length <= maxLength) {
    return str;
  }

  const startLength = Math.ceil(maxLength / 2) - 1;
  const endLength = Math.floor(maxLength / 2) - 1;

  return `${str.substring(0, startLength)}...${str.substring(
    str.length - endLength,
  )}`;
};

interface TxDetailProps {
  show: boolean;
  onHide: () => void;
  link: LinkItem | null;
  currencySymbol?: string;
}

const TxDetail: React.FC<TxDetailProps> = ({
  show,
  onHide,
  link,
  currencySymbol,
}) => {
  // 交易哈希列表来自所选链接参数
  const txHashes = useMemo(() => link?.tx_hash_list || [], [link]);

  // 定义统一的交易详情类型
  interface UnifiedTransactionDetail {
    time: string;
    from: string;
    to: string;
    value: number;
    fee?: number;
    blockHeight?: number;
    txIndex?: number;
    status?: string;
    totalInput?: number;
    totalOutput?: number;
    tx_hash: string;
    timestamp: number;
    from_label?: string;
    to_label?: string;
  }

  const [transactions, setTransactions] = useState<UnifiedTransactionDetail[]>(
    [],
  );
  const [loading, setLoading] = useState<boolean>(false);

  // 复制文本到剪贴板的函数
  const copyToClipboard = (text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        message.success("复制成功");
      })
      .catch((err) => {
        console.error("Failed to copy: ", err);
      });
  };

  // 获取交易详情
  useEffect(() => {
    if (show && link) {
      const fetchTransactionDetails = async () => {
        setLoading(true);
        try {
          // 并行获取所有交易详情
          const transactionPromises = txHashes.map(async (txHash) => {
            // 根据交易哈希判断是哪种类型的交易（这里假设以太坊交易以0x开头）
            if (txHash.startsWith("0x")) {
              try {
                const response =
                  await transactionApi.getEthereumTransactionDetail(txHash);
                if (response.success && response.data) {
                  const ethData = response.data;
                  const txObj = ethData.transaction;
                  return {
                    time: ethData.blockTime,
                    from: txObj.fromAddress,
                    to: txObj.toAddress,
                    value: ethData.value,
                    fee: ethData.fee,
                    blockHeight: ethData.blockHeight,
                    txIndex: txObj.txIndex,
                    status: txObj.status,
                    totalInput: txObj.totalInput,
                    totalOutput: txObj.totalOutput,
                    tx_hash: txObj.txHash,
                    timestamp: new Date(txObj.blockTime).getTime() / 1000,
                  };
                }
              } catch (error) {
                console.error(`获取以太坊交易详情失败 ${txHash}:`, error);
              }
            } else {
              try {
                const response =
                  await transactionApi.getBitcoinTransactionDetail(txHash);
                if (response.success && response.data) {
                  const btcData = response.data;
                  const txObj = btcData.transaction;
                  return {
                    time: btcData.blockTime,
                    from: btcData.fromAddress || txObj.fromAddress || "Unknown",
                    to: btcData.toAddress || txObj.toAddress || "Unknown",
                    value: btcData.value,
                    tx_hash: txObj.txHash,
                    timestamp: new Date(txObj.blockTime).getTime() / 1000,
                  };
                }
              } catch (error) {
                console.error(`获取比特币交易详情失败 ${txHash}:`, error);
              }
            }
            return null;
          });

          const results = await Promise.all(transactionPromises);
          const validTransactions = results.filter(
            (result) => result !== null,
          ) as UnifiedTransactionDetail[];

          setTransactions(validTransactions);
        } catch (error) {
          console.error("获取交易详情出错:", error);
          message.error("获取交易详情失败，请稍后重试");
        } finally {
          setLoading(false);
        }
      };

      fetchTransactionDetails();
    }
  }, [show, link, txHashes]);

  // 如果没有选中的边，返回空内容
  if (!link) {
    return null;
  }

  // 生成交易数据列表
  const transactionData = transactions.map((tx, index) => ({
    key: index,
    time: tx.time,
    from: tx.from,
    to: tx.to,
    value: tx.value,
    fee: tx.fee,
    blockHeight: tx.blockHeight,
    status: tx.status,
    totalInput: tx.totalInput,
    totalOutput: tx.totalOutput,
    tx_hash: tx.tx_hash,
  }));

  // 定义表格列
  const columns: TableProps["columns"] = [
    {
      title: "交易时间",
      dataIndex: "time",
      key: "time",
      width: 180,
      sorter: (a: any, b: any) =>
        new Date(a.time).getTime() - new Date(b.time).getTime(),
      sortDirections: ["descend", "ascend"],
      showSorterTooltip: false,
      render: (time: string) => {
        const date = new Date(time);
        const utc8Date = new Date(date.getTime() + 8 * 60 * 60 * 1000);
        const formatted = utc8Date.toISOString().slice(0, 19).replace("T", " ");
        return <span>{formatted}</span>;
      },
    },
    {
      title: "发送地址",
      dataIndex: "from",
      key: "from",
      render: (text: string) => (
        <Tooltip title={text}>
          <Space align="center">
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "12px",
                display: "inline-block",
                maxWidth: 150,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                verticalAlign: "middle",
              }}
            >
              {truncateMiddle(text)}
            </span>
            <Tooltip title="复制">
              <CopyOutlined
                onClick={() => copyToClipboard(text)}
                style={{
                  cursor: "pointer",
                  color: "#667eea",
                  verticalAlign: "middle",
                }}
              />
            </Tooltip>
          </Space>
        </Tooltip>
      ),
    },
    {
      title: "接收地址",
      dataIndex: "to",
      key: "to",
      render: (text: string) => (
        <Tooltip title={text}>
          <Space align="center">
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "12px",
                display: "inline-block",
                maxWidth: 150,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                verticalAlign: "middle",
              }}
            >
              {truncateMiddle(text)}
            </span>
            <Tooltip title="复制">
              <CopyOutlined
                onClick={() => copyToClipboard(text)}
                style={{
                  cursor: "pointer",
                  color: "#667eea",
                  verticalAlign: "middle",
                }}
              />
            </Tooltip>
          </Space>
        </Tooltip>
      ),
    },
    {
      title: "数额",
      dataIndex: "value",
      key: "value",
      width: 100,
      sorter: (a: any, b: any) => parseFloat(a.value) - parseFloat(b.value),
      sortDirections: ["descend", "ascend"],
      showSorterTooltip: false,
      render: (value: number) => {
        const displayCurrency = currencySymbol || "BTC";
        return (
          <Tag color="#667eea">
            {value} {displayCurrency}
          </Tag>
        );
      },
    },
    {
      title: "手续费",
      dataIndex: "fee",
      key: "fee",
      width: 90,
      render: (fee: number) =>
        fee ? <Tag color="orange">{fee} ETH</Tag> : <span>-</span>,
    },
    {
      title: "区块高度",
      dataIndex: "blockHeight",
      key: "blockHeight",
      width: 100,
      render: (height: number) =>
        height ? <Tag color="blue">{height}</Tag> : <span>-</span>,
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 80,
      render: (status: string) => {
        const statusMap: Record<string, string> = {
          confirmed: "已确认",
          pending: "pending",
          failed: "失败",
        };
        const colorMap: Record<string, string> = {
          confirmed: "green",
          pending: "gold",
          failed: "red",
        };
        return status ? (
          <Tag color={colorMap[status] || "default"}>
            {statusMap[status] || status}
          </Tag>
        ) : (
          <span>-</span>
        );
      },
    },
    {
      title: "交易Hash",
      dataIndex: "tx_hash",
      key: "tx_hash",
      render: (text: string) => (
        <Tooltip title={text}>
          <Space align="center">
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "12px",
                display: "inline-block",
                maxWidth: 150,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                verticalAlign: "middle",
              }}
            >
              {truncateMiddle(text)}
            </span>
            <Tooltip title="复制">
              <CopyOutlined
                onClick={() => copyToClipboard(text)}
                style={{
                  cursor: "pointer",
                  color: "#667eea",
                  verticalAlign: "middle",
                }}
              />
            </Tooltip>
          </Space>
        </Tooltip>
      ),
    },
  ];

  return (
    <ConfigProvider
      theme={{
        components: {
          Table: {
            headerBg: "#f8f9fa",
            headerColor: "#000000",
            bodySortBg: "#ffffff",
            rowHoverBg: "#f5f5f5",
          },
        },
      }}
    >
      <Modal show={show} onHide={onHide} size="lg" dialogClassName="modal-90w">
        <Modal.Header closeButton>
          <Modal.Title>交易明细</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div>
            <Spin spinning={loading} tip="加载交易详情中...">
              <Table
                dataSource={transactionData}
                columns={columns}
                pagination={{ pageSize: 10 }}
                scroll={{ x: "max-content", y: 400 }}
                rowClassName="table-row-center"
              />
            </Spin>
          </div>
        </Modal.Body>
      </Modal>
    </ConfigProvider>
  );
};

export default TxDetail;
