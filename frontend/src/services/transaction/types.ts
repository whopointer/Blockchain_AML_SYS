// 图谱分析相关接口定义
export interface GraphAnalysisNode {
  address: string;
  label?: string;
  type?: string;
  value?: number;
  malicious?: number;
  image?: string;
}

export interface GraphAnalysisEdge {
  from: string;
  to: string;
  value: number;
  timestamp: number;
  tx_hash: string;
  label?: string;
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

export interface BitcoinInput {
  id: number;
  chain: string;
  inputIndex: number;
  prevTxHash: string;
  prevOutIndex: number;
  address: string;
  value: number;
  scriptSig: string;
  createdAt: string;
}

export interface BitcoinOutput {
  id: number;
  chain: string;
  outputIndex: number;
  address: string;
  value: number;
  scriptPubKey: string;
  spentTxHash: string | null;
  spentTime: string | null;
  createdAt: string;
}

export interface BitcoinTransactionDetail {
  outputs: BitcoinOutput[];
  outputCount: number;
  totalOutput: number;
  inputs: BitcoinInput[];
  totalInput: number;
  transaction: {
    id: number;
    chain: string;
    txHash: string;
    blockHeight: number;
    blockTime: string;
    totalInput: number;
    totalOutput: number;
    fee: number;
    txIndex: number | null;
    status: string;
    createdAt: string;
    fromAddress: string | null;
    toAddress: string | null;
    sizeBytes: number;
    locktime: number;
    gasPrice: number | null;
    gasUsed: number | null;
    inputData: string | null;
    valueWei: string | null;
    feeAsDouble: number;
    totalInputAsDouble: number;
    totalOutputAsDouble: number;
  };
  inputCount: number;
}

export interface SingleTransactionDetailResponse<T> {
  success: boolean;
  msg: string;
  data: T;
}

export interface BTCNhopNode {
  id: string;
  label: string;
  title: string;
  addr?: string;
  layer?: number;
  blockHeight?: number;
  time?: string;
  type?: string;
  txHash?: string;
}

export interface BTCNhopEdge {
  val: number;
  tx_hash_list: string[];
  tx_time: string;
  from: string;
  to: string;
  label: string;
}

export interface BTCNhopData {
  tx_count: number;
  address_first_tx_time: string;
  address_latest_tx_time: string;
  latest_tx_time: string;
  node_list: BTCNhopNode[];
  first_tx_time: string;
  edge_list: BTCNhopEdge[];
}

export interface BTCNhopResponse {
  success: boolean;
  message: string;
  data: BTCNhopData;
  total: number;
  code: number;
}
