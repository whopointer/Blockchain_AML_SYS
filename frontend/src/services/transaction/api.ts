import axios from "axios";
import {
  GraphAnalysisResponse,
  EthereumTransactionDetail,
  BitcoinTransactionDetail,
  SingleTransactionDetailResponse,
  BTCNhopResponse,
  BTCPathResponse,
} from "./types";

const TRANSACTION_API_BASE_URL = "http://localhost:8080/api";

const transactionApiClient = axios.create({
  baseURL: TRANSACTION_API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

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

  // BTC N-hop 图谱检索
  getBTCNhopGraph: (
    address: string,
    maxHops: number,
  ): Promise<BTCNhopResponse> =>
    transactionApiClient
      .get(`/neo4j/btc/hops?address=${address}&maxHops=${maxHops}`)
      .then((response) => response.data),

  // BTC 路径追踪
  getBTCPath: (
    fromAddress: string,
    toAddress: string,
  ): Promise<BTCPathResponse> =>
    transactionApiClient
      .get(`/neo4j/btc/path?fromAddress=${fromAddress}&toAddress=${toAddress}`)
      .then((response) => response.data),
};

export default transactionApi;
