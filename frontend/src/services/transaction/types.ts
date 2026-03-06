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
