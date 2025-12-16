import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:5001/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PredictionRequest {
  tx_ids: string[];
}

export interface PredictionResult {
  tx_id: string;
  is_suspicious: boolean;
  confidence_score: number;
  risk_level: 'low' | 'medium' | 'high';
}

export interface PredictionResponse {
  results: PredictionResult[];
  total_transactions: number;
  suspicious_count: number;
  timestamp: string;
}

export interface ModelInfo {
  model_type: string;
  model_version: string;
  loaded_at: string;
  performance_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  model_loaded: boolean;
}

export interface StatisticsResponse {
  system_status: string;
  model_loaded: boolean;
  timestamp: string;
  version: string;
}

export const api = {
  // 健康检查
  healthCheck: (): Promise<HealthResponse> => 
    apiClient.get('/health').then(response => response.data),

  // 单个或多个交易预测
  predictTransactions: (request: PredictionRequest): Promise<PredictionResponse> => 
    apiClient.post('/predict', request).then(response => response.data),

  // 批量预测
  batchPredict: (): Promise<any> => 
    apiClient.post('/batch_predict').then(response => response.data),

  // 获取模型信息
  getModelInfo: (): Promise<ModelInfo> => 
    apiClient.get('/model/info').then(response => response.data),

  // 加载模型
  loadModel: (): Promise<any> => 
    apiClient.post('/model/load').then(response => response.data),

  // 获取统计信息
  getStatistics: (): Promise<StatisticsResponse> => 
    apiClient.get('/statistics').then(response => response.data),

  // 获取预测摘要
  getPredictionSummary: (results: PredictionResult[]): Promise<any> => 
    apiClient.post('/summary', { results }).then(response => response.data),
};

export default api;