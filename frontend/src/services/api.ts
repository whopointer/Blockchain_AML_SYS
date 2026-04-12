

import axios from 'axios';

// 开发阶段：直接连接后端 (不使用网关)
// 可选值: 'http://localhost:8001' (GNN) 或 'http://localhost:8002' (LG-VGAE)
const API_BASE_URL = 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

/** MFTracer HTTP API（与 GNN 预测服务分离），默认端口 8888；可通过 REACT_APP_TRACER_ORIGIN 覆盖 */
function readTracerOriginFromEnv(): string | undefined {
  const g = globalThis as unknown as { process?: { env?: Record<string, string | undefined> } };
  return g.process?.env?.REACT_APP_TRACER_ORIGIN;
}

/**
 * 开发环境下默认走同源相对路径 `/api/v1`，配合 package.json 的 proxy 转发到 MFTracer，避免 CORS。
 * 若已配置 REACT_APP_TRACER_ORIGIN（且后端正确返回 CORS 头），则始终使用该地址。
 */
function getTracerOriginAndApiBase(): { origin: string; apiBase: string } {
  const fromEnv = readTracerOriginFromEnv();
  if (fromEnv && fromEnv.trim() !== '') {
    const origin = fromEnv.replace(/\/$/, '');
    return { origin, apiBase: `${origin}/api/v1` };
  }
  if (process.env.NODE_ENV === 'development') {
    return { origin: '', apiBase: '/api/v1' };
  }
  return { origin: 'http://localhost:8888', apiBase: 'http://localhost:8888/api/v1' };
}

const { origin: TRACER_ORIGIN, apiBase: TRACER_API_BASE_URL } = getTracerOriginAndApiBase();

const tracerApiClient = axios.create({
  baseURL: TRACER_API_BASE_URL,
  timeout: 300000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export { TRACER_ORIGIN };

// ---------- MFTracer：按时间追踪与任务查询 ----------

export interface TraceByTimeRequest {
  start_time: string;
  end_time: string;
  token: string;
  src: string[];
  allowed?: string[];
  forbidden?: string[];
  out_degree_limit?: number;
  depth?: number;
  activate_threshold?: number;
  age_limit?: number;
  label_limit?: number;
}

export interface TraceTaskNode {
  address: string;
  in_flow_usd?: number;
  out_flow_usd?: number;
  net_flow_usd?: number;
  is_source?: boolean;
  is_exchange?: boolean;
  tags?: unknown[];
}

export interface TraceTaskEdge {
  from: string;
  to: string;
  tx_hash?: string;
  block_id?: number;
  amount?: string;
  amount_usd?: number;
  token?: string;
  timestamp?: string;
  age?: number;
}

export interface TraceTaskResult {
  request_id: string;
  status?: string;
  meta?: {
    start_block_id?: number;
    end_block_id?: number;
    token?: string;
    token_address?: string;
    activate_threshold_usd?: number;
    age_limit?: number;
    execution_time_ms?: number;
  };
  summary?: {
    source_address_count?: number;
    node_count?: number;
    edge_count?: number;
    total_flow_usd?: number;
  };
  nodes?: TraceTaskNode[];
  edges?: TraceTaskEdge[];
  files?: Record<string, string>;
}

export interface TraceTaskStatus {
  request_id: string;
  status: string;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string;
  result?: TraceTaskResult;
}

export interface TraceVisualizeRequest {
  request_id: string;
  simplify?: boolean;
  min_value?: number;
  max_nodes?: number;
  layout?: string;
}

export interface TraceVisualizeResponse {
  request_id: string;
  status: string;
  svg_url?: string;
  message?: string;
  error?: string;
}

export interface PredictionRequest {
  tx_ids: string[];
  model_type?: string;  // 可选：指定模型类型
}

export interface PredictionResult {
  tx_id: string;
  is_suspicious: boolean;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high';
}

export interface PredictionResponse {
  results: PredictionResult[];
  model_type: string;  // 返回当前使用的模型类型
  total_transactions: number;
  suspicious_count: number;
  timestamp: string;
}

export interface ModelInfo {
  model_type: string;
  model_version?: string;
  loaded_at?: string;
  performance_metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    auc?: number;
    average_precision?: number;
    threshold?: number;
  };
}

export interface HealthResponse {
  status: string;
  model_type: string;  // 当前模型类型
  timestamp: string;
  model_loaded: boolean;
  data_loaded: boolean;
  cache_built: boolean;
}

export interface StatisticsResponse {
  system_status: string;
  model_type: string;
  model_loaded: boolean;
  model_info?: ModelInfo;
  timestamp: string;
  version: string;
}

// 支持的模型类型
export interface SupportedModel {
  id: string;
  name: string;
  description: string;
}

// 模型列表响应
export interface ModelListResponse {
  supported_models: string[];
  descriptions: {
    [key: string]: string;
  };
}

// 模型切换响应
export interface ModelSwitchResponse {
  success: boolean;
  message: string;
  model_info: ModelInfo;
}

export const api = {
  // 健康检查
  healthCheck: (): Promise<HealthResponse> => 
    apiClient.get('/health').then(response => response.data),

  // 获取支持的模型列表
  getModels: (): Promise<ModelListResponse> => 
    apiClient.get('/models').then(response => response.data),

  // 单个或多个交易预测
  predictTransactions: (request: PredictionRequest): Promise<PredictionResponse> => 
    apiClient.post('/predict', request).then(response => response.data),

  // 指定模型预测
  predictWithModel: (txIds: string[], modelType: string): Promise<PredictionResponse> => 
    apiClient.post('/predict', { tx_ids: txIds, model_type: modelType }).then(response => response.data),

  // 获取模型信息
  getModelInfo: (): Promise<ModelInfo> => 
    apiClient.get('/model/info').then(response => response.data),

  // 切换模型
  switchModel: (modelType: string, experimentName?: string): Promise<ModelSwitchResponse> => 
    apiClient.post('/model/switch', { model_type: modelType, experiment_name: experimentName }).then(response => response.data),

  // 批量检测地址
  batchDetect: (request: BatchDetectRequest): Promise<BatchDetectResponse> =>
    apiClient.post('/batch_detect', request).then(response => response.data),

  // 获取统计信息
  getStatistics: (): Promise<StatisticsResponse> => 
    apiClient.get('/statistics').then(response => response.data),

  // 获取预测摘要
  getPredictionSummary: (results: PredictionResult[]): Promise<any> => 
    apiClient.post('/summary', { results }).then(response => response.data),

  // 洗钱路径追踪
  traceMoneyLaundering: (txId: string, maxDepth: number): Promise<any> => 
    apiClient.post('/trace', { tx_id: txId, max_depth: maxDepth }).then(response => response.data),

  /** MFTracer：按时间范围异步追踪（POST /api/v1/trace-by-time） */
  createTraceTask: (request: TraceByTimeRequest): Promise<TraceTaskStatus> =>
    tracerApiClient.post<TraceTaskStatus>('/trace-by-time', request).then((r) => r.data),

  getTraceTaskStatus: (requestId: string): Promise<TraceTaskStatus> =>
    tracerApiClient.get<TraceTaskStatus>(`/tasks/${encodeURIComponent(requestId)}`).then((r) => r.data),

  generateTraceVisualization: (body: TraceVisualizeRequest): Promise<TraceVisualizeResponse> =>
    tracerApiClient.post<TraceVisualizeResponse>('/visualize', body).then((r) => r.data),

  /** svg_url 为绝对路径时拼接 Tracer  origin；已为完整 URL 则直接使用 */
  getTraceSvgContent: (svgPath: string): Promise<string> => {
    const url =
      svgPath.startsWith('http://') || svgPath.startsWith('https://')
        ? svgPath
        : `${TRACER_ORIGIN}${svgPath.startsWith('/') ? svgPath : `/${svgPath}`}`;
    return axios.get<string>(url, { responseType: 'text', timeout: 120000 }).then((r) => r.data);
  },
};

// 工具函数：获取模型显示名称
export function getModelDisplayName(modelType: string): string {
  const modelNames: { [key: string]: string } = {
    'gnn': 'DGI + GIN + Random Forest',
  };
  return modelNames[modelType.toLowerCase()] || modelType;
}

// 工具函数：获取模型颜色
export function getModelColor(modelType: string): string {
  const modelColors: { [key: string]: string } = {
    'gnn': '#667eea',
  };
  return modelColors[modelType.toLowerCase()] || '#6c757d';
}

// ============================================================
// 任务相关接口
// ============================================================

// 创建任务请求
export interface CreateTaskRequest {
  address: string;
  address_type?: string;
  neighbor_blocks?: number;
  external_ref?: string;
  submitted_by?: string;
}

// 任务响应
export interface TaskResponse {
  task_id: string;
  status: string;
  address: string;
  address_type: string;
  created_at: string;
  message?: string;
}

// 任务详情
export interface TaskDetail {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  address: string;
  address_type: string;
  probability?: number;
  risk_label?: string;
  is_suspicious?: boolean;
  features?: Record<string, number>;
  result?: TaskResultDetail;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

// 任务结果详情
export interface TaskResultDetail {
  task_id: string;
  gnn_probability?: number;
  gnn_is_suspicious?: boolean;
  final_label?: string;
  confidence?: number;
  risk_level?: string;
  model_version?: string;
  threshold_used?: number;
}

// 任务列表响应
export interface TaskListResponse {
  tasks: TaskDetail[];
  total: number;
  limit: number;
  offset: number;
}

// 检测请求接口
export interface DetectRequest {
  address: string;
  address_type?: string;
  model_type?: string;
  neighbor_depth?: number;
}

// 检测响应接口
export interface DetectResponse {
  address: string;
  address_type: string;
  model_type: string;
  probability: number;
  is_suspicious: boolean;
  risk_label: string;
  original_label?: string;
  subgraph_info?: {
    total_nodes: number;
    total_edges: number;
    neighbor_depth: number;
  };
  timestamp: string;
}

// 批量检测请求接口
export interface BatchDetectRequest {
  addresses: string[];
  address_type?: string;
  model_type?: string;
  neighbor_depth?: number;
}

// 批量检测响应接口
export interface BatchDetectResponse {
  results: DetectResponse[];
  statistics: {
    total: number;
    success: number;
    error: number;
    suspicious: number;
    normal: number;
  };
  model_type: string;
  timestamp: string;
}

// 同步检测 API（简化版）
export const detectionApi = {
  // 同步检测地址风险
  detect: (request: DetectRequest): Promise<DetectResponse> =>
    apiClient.post('/detect', request).then(response => response.data),

  // 批量检测地址风险（含详细信息）
  batchDetect: (request: BatchDetectRequest): Promise<BatchDetectResponse> =>
    apiClient.post('/batch_detect', request).then(response => response.data),
};

// 任务 API（已废弃，推荐使用 detectionApi）
export const taskApi = {
  // 创建任务（已废弃）
  createTask: (request: CreateTaskRequest): Promise<TaskResponse> =>
    apiClient.post('/tasks', request).then(response => response.data),

  // 获取任务详情
  getTask: (taskId: string): Promise<TaskDetail> =>
    apiClient.get(`/tasks/${taskId}`).then(response => response.data),

  // 获取任务列表
  listTasks: (params?: {
    status?: string;
    address?: string;
    limit?: number;
    offset?: number;
  }): Promise<TaskListResponse> =>
    apiClient.get('/tasks', { params }).then(response => response.data),

  // 删除任务
  deleteTask: (taskId: string): Promise<{ message: string }> =>
    apiClient.delete(`/tasks/${taskId}`).then(response => response.data),
};

// 工具函数：获取风险等级颜色
export function getRiskLevelColor(riskLevel?: string): string {
  const colors: { [key: string]: string } = {
    'high': '#dc3545',
    'medium': '#fd7e14',
    'low': '#ffc107',
    'normal': '#28a745',
  };
  return colors[riskLevel?.toLowerCase() || ''] || '#6c757d';
}

// 工具函数：获取任务状态颜色
export function getTaskStatusColor(status?: string): string {
  const colors: { [key: string]: string } = {
    'pending': '#6c757d',
    'processing': '#0dcaf0',
    'completed': '#28a745',
    'failed': '#dc3545',
  };
  return colors[status?.toLowerCase() || ''] || '#6c757d';
}

export default api;
