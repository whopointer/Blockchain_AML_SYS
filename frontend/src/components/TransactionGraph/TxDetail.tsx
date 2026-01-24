import React from "react";
import { Modal } from "react-bootstrap";
import {
  Table,
  Tag,
  TableProps,
  Space,
  Tooltip,
  message,
  ConfigProvider,
} from "antd";
import { CopyOutlined } from "@ant-design/icons";
import { LinkItem } from "./types";
import batchTxHashDetail from "./batch_tx_hash_detail.json";

// 自定义函数用于在文本中间添加省略号
const truncateMiddle = (str: string, maxLength: number = 15) => {
  if (str.length <= maxLength) {
    return str;
  }

  const startLength = Math.ceil(maxLength / 2) - 1;
  const endLength = Math.floor(maxLength / 2) - 1;

  return `${str.substring(0, startLength)}...${str.substring(
    str.length - endLength
  )}`;
};

interface TxDetailProps {
  show: boolean;
  onHide: () => void;
  link: LinkItem | null;
}

const TxDetail: React.FC<TxDetailProps> = ({ show, onHide, link }) => {
  // 如果没有选中的边，返回空内容
  if (!link) {
    return null;
  }

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

  const filteredTransactions = batchTxHashDetail.tx_detail_list;

  // 生成交易数据列表
  const transactionData = filteredTransactions.map((tx, index) => ({
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
          Tooltip: {
            colorTextLightSolid: "#000000",
          },
          Table: {
            headerBg: "#244963",
            headerColor: "#ffffff",
            bodySortBg: "#1a3a52",
            rowHoverBg: "#3a5f7f",
          },
        },
      }}
    >
      <Modal
        show={show}
        onHide={onHide}
        size="lg"
        dialogClassName="modal-90w"
        contentClassName="dark-modal"
      >
        <Modal.Header
          closeButton
          style={{ backgroundColor: "#244963", borderColor: "#3a5f7f" }}
        >
          <Modal.Title style={{ color: "#ffffff" }}>交易明细</Modal.Title>
        </Modal.Header>
        <Modal.Body style={{ backgroundColor: "#1a3a52", color: "#d8e3f0" }}>
          <div>
            <Table
              dataSource={transactionData}
              columns={columns}
              pagination={{ pageSize: 10 }}
              scroll={{ y: 400 }}
            />
          </div>
        </Modal.Body>
      </Modal>
    </ConfigProvider>
  );
};

export default TxDetail;
