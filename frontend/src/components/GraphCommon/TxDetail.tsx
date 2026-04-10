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
  Input,
  Card,
  Row,
  Col,
  Descriptions,
} from "antd";
import { CopyOutlined, SearchOutlined } from "@ant-design/icons";
import { LinkItem } from "./types";
import { transactionApi } from "@/services/transaction";
import {
  BitcoinTransactionDetail,
  BitcoinInput,
  BitcoinOutput,
} from "@/services/transaction/types";

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
  const txHashes = useMemo(() => link?.tx_hash_list || [], [link]);

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
  const [btcTransactionDetail, setBtcTransactionDetail] =
    useState<BitcoinTransactionDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [searchText, setSearchText] = useState<string>("");
  const [debouncedSearchText, setDebouncedSearchText] = useState<string>("");
  const [inputSearch, setInputSearch] = useState<string>("");
  const [debouncedInputSearch, setDebouncedInputSearch] = useState<string>("");
  const [outputSearch, setOutputSearch] = useState<string>("");
  const [debouncedOutputSearch, setDebouncedOutputSearch] =
    useState<string>("");

  const isBitcoin = useMemo(() => {
    if (txHashes.length === 0) return false;
    return !txHashes[0].startsWith("0x");
  }, [txHashes]);

  // 防抖处理
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchText(searchText);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchText]);

  // 输入搜索防抖
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedInputSearch(inputSearch);
    }, 300);

    return () => clearTimeout(timer);
  }, [inputSearch]);

  // 输出搜索防抖
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedOutputSearch(outputSearch);
    }, 300);

    return () => clearTimeout(timer);
  }, [outputSearch]);

  // 弹窗关闭时清空搜索框
  useEffect(() => {
    if (!show) {
      setSearchText("");
      setDebouncedSearchText("");
      setBtcTransactionDetail(null);
      setInputSearch("");
      setDebouncedInputSearch("");
      setOutputSearch("");
      setDebouncedOutputSearch("");
    }
  }, [show]);

  // 过滤交易数据
  const filteredTransactions = useMemo(() => {
    if (!debouncedSearchText.trim()) {
      return transactions;
    }
    const searchLower = debouncedSearchText.toLowerCase();
    return transactions.filter(
      (tx) =>
        tx.from.toLowerCase().includes(searchLower) ||
        tx.to.toLowerCase().includes(searchLower) ||
        tx.tx_hash.toLowerCase().includes(searchLower),
    );
  }, [transactions, debouncedSearchText]);

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
        setBtcTransactionDetail(null);

        if (txHashes.length > 0 && !txHashes[0].startsWith("0x")) {
          try {
            const response = await transactionApi.getBitcoinTransactionDetail(
              txHashes[0],
            );
            if (response.success && response.data) {
              setBtcTransactionDetail(response.data);
              const btcData = response.data;
              const txObj = btcData.transaction;
              const inputAddresses = btcData.inputs.map(
                (input: BitcoinInput) => input.address,
              );
              const outputAddresses = btcData.outputs.map(
                (output: BitcoinOutput) => output.address,
              );
              const uniqueInputs = [...new Set(inputAddresses)];
              const uniqueOutputs = [...new Set(outputAddresses)];

              setTransactions([
                {
                  time: txObj.blockTime,
                  from: uniqueInputs.join(", ") || "Unknown",
                  to: uniqueOutputs.join(", ") || "Unknown",
                  value: txObj.totalOutput,
                  fee: txObj.fee,
                  blockHeight: txObj.blockHeight,
                  status: txObj.status,
                  totalInput: txObj.totalInput,
                  totalOutput: txObj.totalOutput,
                  tx_hash: txObj.txHash,
                  timestamp: new Date(txObj.blockTime).getTime() / 1000,
                },
              ]);
            }
          } catch (error) {
            console.error(`获取比特币交易详情失败 ${txHashes[0]}:`, error);
            message.error("获取交易详情失败，请稍后重试");
          } finally {
            setLoading(false);
          }
          return;
        }

        try {
          const transactionPromises = txHashes.map(async (txHash) => {
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
                  const inputAddresses = btcData.inputs.map(
                    (input: BitcoinInput) => input.address,
                  );
                  const outputAddresses = btcData.outputs.map(
                    (output: BitcoinOutput) => output.address,
                  );
                  return {
                    time: txObj.blockTime,
                    from: [...new Set(inputAddresses)].join(", ") || "Unknown",
                    to: [...new Set(outputAddresses)].join(", ") || "Unknown",
                    value: txObj.totalOutput,
                    fee: txObj.fee,
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
  const transactionData = filteredTransactions.map((tx, index) => ({
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

  const inputColumns: TableProps["columns"] = [
    {
      title: "输入索引",
      dataIndex: "inputIndex",
      key: "inputIndex",
      width: 80,
      render: (index: number) => <Tag color="blue">{index}</Tag>,
    },
    {
      title: "前一交易哈希",
      dataIndex: "prevTxHash",
      key: "prevTxHash",
      width: 200,
      render: (text: string) => (
        <Tooltip title={text}>
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "11px",
              display: "inline-block",
              maxWidth: 180,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {truncateMiddle(text, 20)}
          </span>
        </Tooltip>
      ),
    },
    {
      title: "前一输出索引",
      dataIndex: "prevOutIndex",
      key: "prevOutIndex",
      width: 100,
      render: (index: number) => <Tag>{index}</Tag>,
    },
    {
      title: "地址",
      dataIndex: "address",
      key: "address",
      width: 250,
      render: (text: string) => (
        <Tooltip title={text}>
          <Space align="center">
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "11px",
                display: "inline-block",
                maxWidth: 200,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                verticalAlign: "middle",
              }}
            >
              {truncateMiddle(text, 25)}
            </span>
            <Tooltip title="复制">
              <CopyOutlined
                onClick={() => copyToClipboard(text)}
                style={{
                  cursor: "pointer",
                  color: "#667eea",
                  verticalAlign: "middle",
                  fontSize: "12px",
                }}
              />
            </Tooltip>
          </Space>
        </Tooltip>
      ),
    },
    {
      title: "金额 (BTC)",
      dataIndex: "value",
      key: "value",
      width: 120,
      render: (value: number) => <Tag color="green">{value}</Tag>,
    },
  ];

  const outputColumns: TableProps["columns"] = [
    {
      title: "输出索引",
      dataIndex: "outputIndex",
      key: "outputIndex",
      width: 80,
      render: (index: number) => <Tag color="blue">{index}</Tag>,
    },
    {
      title: "地址",
      dataIndex: "address",
      key: "address",
      width: 250,
      render: (text: string) => (
        <Tooltip title={text}>
          <Space align="center">
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "11px",
                display: "inline-block",
                maxWidth: 200,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                verticalAlign: "middle",
              }}
            >
              {truncateMiddle(text, 25)}
            </span>
            <Tooltip title="复制">
              <CopyOutlined
                onClick={() => copyToClipboard(text)}
                style={{
                  cursor: "pointer",
                  color: "#667eea",
                  verticalAlign: "middle",
                  fontSize: "12px",
                }}
              />
            </Tooltip>
          </Space>
        </Tooltip>
      ),
    },
    {
      title: "金额 (BTC)",
      dataIndex: "value",
      key: "value",
      width: 120,
      render: (value: number) => <Tag color="green">{value}</Tag>,
    },
    {
      title: "是否已花费",
      dataIndex: "spentTxHash",
      key: "spentTxHash",
      width: 100,
      render: (spent: string | null) =>
        spent ? (
          <Tag color="red">已花费</Tag>
        ) : (
          <Tag color="default">未花费</Tag>
        ),
    },
  ];

  const formatBlockTime = (time: string) => {
    const date = new Date(time);
    const utc8Date = new Date(date.getTime() + 8 * 60 * 60 * 1000);
    return utc8Date.toISOString().slice(0, 19).replace("T", " ");
  };

  const renderBitcoinDetail = () => {
    if (!btcTransactionDetail) return null;

    const {
      inputs,
      outputs,
      transaction,
      totalInput,
      totalOutput,
      inputCount,
      outputCount,
    } = btcTransactionDetail;
    const statusMap: Record<string, string> = {
      confirmed: "已确认",
      pending: "pending",
      failed: "失败",
    };
    const statusColorMap: Record<string, string> = {
      confirmed: "green",
      pending: "gold",
      failed: "red",
    };

    // 过滤输入数据
    const filteredInputs = debouncedInputSearch.trim()
      ? inputs.filter(
          (input) =>
            input.address
              .toLowerCase()
              .includes(debouncedInputSearch.toLowerCase()) ||
            input.prevTxHash
              .toLowerCase()
              .includes(debouncedInputSearch.toLowerCase()),
        )
      : inputs;

    // 过滤输出数据
    const filteredOutputs = debouncedOutputSearch.trim()
      ? outputs.filter((output) =>
          output.address
            .toLowerCase()
            .includes(debouncedOutputSearch.toLowerCase()),
        )
      : outputs;

    return (
      <div>
        <Card title="交易基本信息" size="small" style={{ marginBottom: 16 }}>
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="交易哈希">
              <Space>
                <span style={{ fontFamily: "monospace", fontSize: "11px" }}>
                  {transaction.txHash}
                </span>
                <CopyOutlined
                  onClick={() => copyToClipboard(transaction.txHash)}
                  style={{ cursor: "pointer", color: "#667eea" }}
                />
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="区块高度">
              <Tag color="blue">{transaction.blockHeight}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="区块时间">
              {formatBlockTime(transaction.blockTime)}
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              <Tag color={statusColorMap[transaction.status] || "default"}>
                {statusMap[transaction.status] || transaction.status}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="输入总额">
              <Tag color="purple">{totalInput} BTC</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="输出总额">
              <Tag color="purple">{totalOutput} BTC</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="手续费">
              <Tag color="orange">{transaction.fee} BTC</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="交易大小">
              {transaction.sizeBytes} bytes
            </Descriptions.Item>
          </Descriptions>
        </Card>

        <Row gutter={16}>
          <Col span={12}>
            <Card
              title={`输入 (${filteredInputs.length}/${inputCount})`}
              size="small"
              style={{ marginBottom: 16 }}
              bodyStyle={{ padding: "12px" }}
            >
              <div style={{ marginBottom: 12 }}>
                <Input
                  placeholder="搜索地址或前一交易哈希"
                  prefix={<SearchOutlined />}
                  value={inputSearch}
                  onChange={(e) => setInputSearch(e.target.value)}
                  size="small"
                  style={{ width: "100%" }}
                />
              </div>
              <Table
                dataSource={filteredInputs.map((item, idx) => ({
                  ...item,
                  key: idx,
                }))}
                columns={inputColumns}
                pagination={false}
                scroll={{ y: 250 }}
                size="small"
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card
              title={`输出 (${filteredOutputs.length}/${outputCount})`}
              size="small"
              style={{ marginBottom: 16 }}
              bodyStyle={{ padding: "12px" }}
            >
              <div style={{ marginBottom: 12 }}>
                <Input
                  placeholder="搜索地址"
                  prefix={<SearchOutlined />}
                  value={outputSearch}
                  onChange={(e) => setOutputSearch(e.target.value)}
                  size="small"
                  style={{ width: "100%" }}
                />
              </div>
              <Table
                dataSource={filteredOutputs.map((item, idx) => ({
                  ...item,
                  key: idx,
                }))}
                columns={outputColumns}
                pagination={false}
                scroll={{ y: 250 }}
                size="small"
              />
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

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
      <Modal show={show} onHide={onHide} size="xl" dialogClassName="modal-90w">
        <Modal.Header closeButton>
          <Modal.Title>交易明细 {isBitcoin ? "(BTC)" : "(ETH)"}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {isBitcoin ? (
            <div style={{ minHeight: 400 }}>
              {loading ? (
                <div
                  style={{
                    minHeight: 400,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                  }}
                >
                  <Spin tip="加载交易详情中..." />
                </div>
              ) : (
                renderBitcoinDetail()
              )}
            </div>
          ) : (
            <>
              <div style={{ marginBottom: 16 }}>
                <Input
                  placeholder="搜索发送地址、接收地址或交易哈希"
                  prefix={<SearchOutlined />}
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  style={{ width: "100%" }}
                />
              </div>
              <div style={{ minHeight: 400 }}>
                {loading ? (
                  <div
                    style={{
                      minHeight: 400,
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                    }}
                  >
                    <Spin tip="加载交易详情中..." />
                  </div>
                ) : (
                  <Table
                    dataSource={transactionData}
                    columns={columns}
                    pagination={false}
                    scroll={{ x: "max-content", y: 400 }}
                    rowClassName="table-row-center"
                  />
                )}
              </div>
            </>
          )}
        </Modal.Body>
      </Modal>
    </ConfigProvider>
  );
};

export default TxDetail;
