import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

function readTracerOriginFromEnv(): string | undefined {
  const g = globalThis as unknown as {
    process?: { env?: Record<string, string | undefined> };
  };
  return g.process?.env?.REACT_APP_TRACER_ORIGIN;
}

function getTracerOriginAndApiBase(): { origin: string; apiBase: string } {
  const fromEnv = readTracerOriginFromEnv();
  if (fromEnv && fromEnv.trim() !== "") {
    const origin = fromEnv.replace(/\/$/, "");
    return { origin, apiBase: `${origin}/api/v1` };
  }
  if (process.env.NODE_ENV === "development") {
    return { origin: "", apiBase: "/api/v1" };
  }
  return {
    origin: "http://localhost:8888",
    apiBase: "http://localhost:8888/api/v1",
  };
}

const { origin: TRACER_ORIGIN, apiBase: TRACER_API_BASE_URL } =
  getTracerOriginAndApiBase();

const tracerApiClient = axios.create({
  baseURL: TRACER_API_BASE_URL,
  timeout: 300000,
  headers: {
    "Content-Type": "application/json",
  },
});

export { TRACER_ORIGIN };

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
  model_type?: string;
}

export interface PredictionResult {
  tx_id: string;
  is_suspicious: boolean;
  confidence: number;
  risk_level: "low" | "medium" | "high";
}

export interface PredictionResponse {
  results: PredictionResult[];
  model_type: string;
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
  model_type: string;
  timestamp: string;
  model_loaded: boolean;
  data_loaded?: boolean;
  cache_built?: boolean;
}

export interface StatisticsResponse {
  system_status: string;
  model_type: string;
  model_loaded: boolean;
  model_info?: ModelInfo;
  timestamp: string;
  version: string;
}

export interface SupportedModel {
  id: string;
  name: string;
  description: string;
}

export interface ModelListResponse {
  supported_models: string[];
  descriptions: {
    [key: string]: string;
  };
}

export interface ModelSwitchResponse {
  success: boolean;
  message: string;
  model_info: ModelInfo;
}

export const api = {
  // 健康检查
  healthCheck: (): Promise<HealthResponse> =>
    apiClient.get("/health").then((response) => response.data),

  // 获取支持的模型列表
  getModels: (): Promise<ModelListResponse> =>
    apiClient.get("/models").then((response) => response.data),

  // 单个或多个交易预测
  predictTransactions: (
    request: PredictionRequest,
  ): Promise<PredictionResponse> =>
    apiClient.post("/predict", request).then((response) => response.data),

  // 指定模型预测
  predictWithModel: (
    txIds: string[],
    modelType: string,
  ): Promise<PredictionResponse> =>
    apiClient
      .post("/predict", { tx_ids: txIds, model_type: modelType })
      .then((response) => response.data),

  // 批量预测
  batchPredict: (modelType?: string): Promise<any> =>
    apiClient
      .post("/batch_predict", null, { params: { model_type: modelType } })
      .then((response) => response.data),

  // 获取模型信息
  getModelInfo: (): Promise<ModelInfo> =>
    apiClient.get("/model/info").then((response) => response.data),

  // 切换模型
  switchModel: (
    modelType: string,
    experimentName?: string,
  ): Promise<ModelSwitchResponse> =>
    apiClient
      .post("/model/switch", {
        model_type: modelType,
        experiment_name: experimentName,
      })
      .then((response) => response.data),

  // 获取统计信息
  getStatistics: (): Promise<StatisticsResponse> =>
    apiClient.get("/statistics").then((response) => response.data),

  // 获取预测摘要
  getPredictionSummary: (results: PredictionResult[]): Promise<any> =>
    apiClient.post("/summary", { results }).then((response) => response.data),

  // 洗钱路径追踪
  traceMoneyLaundering: (txId: string, maxDepth: number): Promise<any> =>
    apiClient
      .post("/trace", { tx_id: txId, max_depth: maxDepth })
      .then((response) => response.data),

  // MFTracer：按时间范围异步追踪
  createTraceTask: (request: TraceByTimeRequest): Promise<TraceTaskStatus> =>
    tracerApiClient
      .post<TraceTaskStatus>("/trace-by-time", request)
      .then((r) => r.data),

  getTraceTaskStatus: (requestId: string): Promise<TraceTaskStatus> =>
    tracerApiClient
      .get<TraceTaskStatus>(`/tasks/${encodeURIComponent(requestId)}`)
      .then((r) => r.data),

  generateTraceVisualization: (
    body: TraceVisualizeRequest,
  ): Promise<TraceVisualizeResponse> =>
    tracerApiClient
      .post<TraceVisualizeResponse>("/visualize", body)
      .then((r) => r.data),

  getTraceSvgContent: (svgPath: string): Promise<string> => {
    const url =
      svgPath.startsWith("http://") || svgPath.startsWith("https://")
        ? svgPath
        : `${TRACER_ORIGIN}${svgPath.startsWith("/") ? svgPath : `/${svgPath}`}`;
    return axios
      .get<string>(url, { responseType: "text", timeout: 120000 })
      .then((r) => r.data);
  },
};

export function getModelDisplayName(modelType: string): string {
  const modelNames: { [key: string]: string } = {
    gnn: "DGI + GIN + Random Forest",
  };
  return modelNames[modelType.toLowerCase()] || modelType;
}

export function getModelColor(modelType: string): string {
  const modelColors: { [key: string]: string } = {
    gnn: "#667eea",
  };
  return modelColors[modelType.toLowerCase()] || "#6c757d";
}

export interface CreateTaskRequest {
  address: string;
  address_type?: string;
  neighbor_blocks?: number;
  external_ref?: string;
  submitted_by?: string;
}

export interface TaskResponse {
  task_id: string;
  status: string;
  address: string;
  address_type: string;
  created_at: string;
  message?: string;
}

export interface TaskDetail {
  task_id: string;
  status: "pending" | "processing" | "completed" | "failed";
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

export interface TaskListResponse {
  tasks: TaskDetail[];
  total: number;
  limit: number;
  offset: number;
}

export interface DetectRequest {
  address: string;
  address_type?: string;
  model_type?: string;
  neighbor_depth?: number;
}

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

export const detectionApi = {
  detect: (request: DetectRequest): Promise<DetectResponse> =>
    apiClient.post("/detect", request).then((response) => response.data),
};

export const taskApi = {
  createTask: (request: CreateTaskRequest): Promise<TaskResponse> =>
    apiClient.post("/tasks", request).then((response) => response.data),

  getTask: (taskId: string): Promise<TaskDetail> =>
    apiClient.get(`/tasks/${taskId}`).then((response) => response.data),

  listTasks: (params?: {
    status?: string;
    address?: string;
    limit?: number;
    offset?: number;
  }): Promise<TaskListResponse> =>
    apiClient.get("/tasks", { params }).then((response) => response.data),

  deleteTask: (taskId: string): Promise<{ message: string }> =>
    apiClient.delete(`/tasks/${taskId}`).then((response) => response.data),
};

export function getRiskLevelColor(riskLevel?: string): string {
  const colors: { [key: string]: string } = {
    high: "#dc3545",
    medium: "#fd7e14",
    low: "#ffc107",
    normal: "#28a745",
  };
  return colors[riskLevel?.toLowerCase() || ""] || "#6c757d";
}

export function getTaskStatusColor(status?: string): string {
  const colors: { [key: string]: string } = {
    pending: "#6c757d",
    processing: "#0dcaf0",
    completed: "#28a745",
    failed: "#dc3545",
  };
  return colors[status?.toLowerCase() || ""] || "#6c757d";
}

// 导出所有API服务
export { transactionApi } from "./transaction";
export { graphSnapshotApi } from "./graph-snapshot";

export default api;
