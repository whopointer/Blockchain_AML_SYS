import axios from "axios";

const TRANSACTION_API_BASE_URL = "http://localhost:8080/api";

const transactionApiClient = axios.create({
  baseURL: TRANSACTION_API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface TransactionDetailRequest {
  tx_hash_list: string[];
}

export interface TransactionDetail {
  time: string;
  from: string;
  to: string;
  value: number;
  tx_hash: string;
  timestamp: number;
  from_label: string;
  to_label: string;
}

export interface TransactionDetailResponse {
  success: boolean;
  msg: string;
  tx_detail_list: TransactionDetail[];
}

export const transactionApi = {
  // 获取交易详情
  getTransactionDetails: (
    request: TransactionDetailRequest,
  ): Promise<TransactionDetailResponse> =>
    transactionApiClient
      .post("/transactions/batch-detail", request)
      .then((response) => response.data),
};

export default transactionApi;
