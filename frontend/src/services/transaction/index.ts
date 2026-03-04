import axios from "axios";

const TRANSACTION_API_BASE_URL = "http://localhost:8080/api";

const transactionApiClient = axios.create({
  baseURL: TRANSACTION_API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// 图谱分析相关接口定义
export interface GraphAnalysisNode {
  address: string;
  label?: string;
  type?: string;
  value?: number;
}

export interface GraphAnalysisEdge {
  from: string;
  to: string;
  value: number;
  timestamp: number;
  tx_hash: string;
}

export interface GraphAnalysisResponse {
  success: boolean;
  msg: string;
  data: {
    node_list: GraphAnalysisNode[];
    edge_list: GraphAnalysisEdge[];
    tx_count: number;
    first_tx_time: number;
    latest_tx_time: number;
    address_first_tx_time: number;
    address_latest_tx_time: number;
  };
}
export interface EthereumTransactionDetail {
  transaction: {
    id: number;
    chain: string;
    txHash: string;
    blockHeight: number;
    blockTime: string; // ISO string
    totalInput: number | null;
    totalOutput: number;
    fee: number;
    txIndex: number;
    status: string;
    createdAt: string;
    fromAddress: string;
    toAddress: string;
    getTotalInputAsDouble: number;
    getTotalOutputAsDouble: number;
    getFeeAsDouble: number;
  };
  fromAddress: string;
  toAddress: string;
  value: number;
  fee: number;
  blockHeight: number;
  blockTime: string;
}

export interface BitcoinTransactionDetail {
  transaction: {
    id: number;
    txHash: string;
    blockHeight: number;
    blockTime: string;
    fee: number;
    fromAddress?: string;
    toAddress?: string;
  };
  fromAddress: string;
  toAddress: string;
  value: number;
  fee: number;
  blockHeight: number;
  blockTime: string;
}

export interface SingleTransactionDetailResponse<T> {
  success: boolean;
  msg: string;
  data: T;
}

export const transactionApi = {
  // 获取以太坊交易详情
  getEthereumTransactionDetail: (
    txHash: string,
  ): Promise<SingleTransactionDetailResponse<EthereumTransactionDetail>> =>
    transactionApiClient
      .get(`/ethereum/transaction/${txHash}`)
      .then((response) => response.data),

  // 获取比特币交易详情
  getBitcoinTransactionDetail: (
    txHash: string,
  ): Promise<SingleTransactionDetailResponse<BitcoinTransactionDetail>> =>
    transactionApiClient
      .get(`/bitcoin/transaction/${txHash}`)
      .then((response) => response.data),

  // N-hop 图谱检索
  getNhopGraph: (
    address: string,
    maxHops: number,
  ): Promise<GraphAnalysisResponse> =>
    transactionApiClient
      .get(`/neo4j/address/hops?address=${address}&maxHops=${maxHops}`)
      .then((response) => response.data),

  // 查找节点间路径
  getAllPath: (
    address1: string,
    address2: string,
  ): Promise<GraphAnalysisResponse> =>
    transactionApiClient
      .get(`/neo4j/path?fromAddress=${address1}&toAddress=${address2}`)
      .then((response) => response.data),
};

export default transactionApi;
