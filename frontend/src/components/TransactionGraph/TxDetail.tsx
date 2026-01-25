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
import {
  TransactionDetail,
  transactionApi,
} from "../../services/transaction/index";

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
}

const TxDetail: React.FC<TxDetailProps> = ({ show, onHide, link }) => {
  // 测试用的交易哈希列表
  const testTxHashes = useMemo(
    () => [
      "0x000109f8e4760f085ddc9df66f891e24b0b219a3d258aba479c7cdb0a9a1cd9f",
      "0x00019497573dedae9387f8b3eb3a4e1a0622eb473f527c3e673811599ba25d86",
    ],
    [],
  );

  const [transactions, setTransactions] = useState<TransactionDetail[]>([]);
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
          const response = await transactionApi.getTransactionDetails({
            tx_hash_list: testTxHashes,
          });
          if (response.success) {
            setTransactions(response.tx_detail_list);
          } else {
            message.error(`获取交易详情失败: ${response.msg}`);
          }
        } catch (error) {
          console.error("获取交易详情出错:", error);
          message.error("获取交易详情失败，请稍后重试");
        } finally {
          setLoading(false);
        }
      };

      fetchTransactionDetails();
    }
  }, [show, link, testTxHashes]);

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
    tx_hash: tx.tx_hash,
  }));

  // 定义表格列
  const columns: TableProps["columns"] = [
    {
      title: "交易时间(UTC)",
      dataIndex: "time",
      key: "time",
      width: 180,
      sorter: (a: any, b: any) =>
        new Date(a.time).getTime() - new Date(b.time).getTime(),
      sortDirections: ["descend", "ascend"],
      showSorterTooltip: false,
    },
    {
      title: "发送地址",
      dataIndex: "from",
      key: "from",
      render: (text: string) => (
        <Space>
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "12px",
              display: "inline-block",
            }}
            title={text}
          >
            {truncateMiddle(text)}
          </span>
          <Tooltip title="复制">
            <CopyOutlined
              onClick={() => copyToClipboard(text)}
              style={{ cursor: "pointer", color: "#667eea" }}
            />
          </Tooltip>
        </Space>
      ),
    },
    {
      title: "接收地址",
      dataIndex: "to",
      key: "to",
      render: (text: string) => (
        <Space>
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "12px",
              display: "inline-block",
            }}
            title={text}
          >
            {truncateMiddle(text)}
          </span>
          <Tooltip title="复制">
            <CopyOutlined
              onClick={() => copyToClipboard(text)}
              style={{ cursor: "pointer", color: "#667eea" }}
            />
          </Tooltip>
        </Space>
      ),
    },
    {
      title: "数额",
      dataIndex: "value",
      key: "value",
      width: 100,
      sorter: (a: any, b: any) => a.value - b.value,
      sortDirections: ["descend", "ascend"],
      showSorterTooltip: false,
      render: (value: number) => <Tag color="#667eea">{value} BNB</Tag>,
    },
    {
      title: "交易Hash",
      dataIndex: "tx_hash",
      key: "tx_hash",
      render: (text: string) => (
        <Space>
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "12px",
              display: "inline-block",
            }}
            title={text}
          >
            {truncateMiddle(text)}
          </span>
          <Tooltip title="复制">
            <CopyOutlined
              onClick={() => copyToClipboard(text)}
              style={{ cursor: "pointer", color: "#667eea" }}
            />
          </Tooltip>
        </Space>
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
                scroll={{ y: 400 }}
              />
            </Spin>
          </div>
        </Modal.Body>
      </Modal>
    </ConfigProvider>
  );
};

export default TxDetail;
