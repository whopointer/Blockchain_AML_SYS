import axios from "axios";

const API_BASE_URL = "http://localhost:8080/api";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// 案件管理 API
export const caseApi = {
  // 创建案件
  createCase: (data: any) =>
    apiClient.post("/cases", data).then((res) => res.data),

  // 获取所有案件
  getAllCases: () => apiClient.get("/cases").then((res) => res.data),

  // 根据ID获取案件
  getCaseById: (id: string) =>
    apiClient.get(`/cases/${id}`).then((res) => res.data),

  // 根据编号获取案件
  getCaseByNumber: (caseNumber: string) =>
    apiClient.get(`/cases/number/${caseNumber}`).then((res) => res.data),

  // 更新案件
  updateCase: (id: string, data: any) =>
    apiClient.put(`/cases/${id}`, data).then((res) => res.data),

  // 删除案件
  deleteCase: (id: string) =>
    apiClient.delete(`/cases/${id}`).then((res) => res.data),

  // 按状态查询
  getCasesByStatus: (status: string) =>
    apiClient.get(`/cases/status/${status}`).then((res) => res.data),

  // 按风险等级查询
  getCasesByRiskLevel: (riskLevel: string) =>
    apiClient.get(`/cases/risk/${riskLevel}`).then((res) => res.data),

  // 搜索案件
  searchCases: (keyword: string) =>
    apiClient.get(`/cases/search?keyword=${keyword}`).then((res) => res.data),

  // 更新案件状态
  updateCaseStatus: (id: string, status: string) =>
    apiClient
      .put(`/cases/${id}/status?status=${status}`)
      .then((res) => res.data),

  // 指派案件
  assignCase: (id: string, assignedTo: string) =>
    apiClient
      .put(`/cases/${id}/assign?assignedTo=${assignedTo}`)
      .then((res) => res.data),
};

// 订阅管理 API
export const subscriptionApi = {
  // ========== 节点订阅 ==========

  // 创建节点订阅
  createNodeSubscription: (data: any) =>
    apiClient.post("/subscriptions/nodes", data).then((res) => res.data),

  // 获取所有节点订阅
  getAllNodeSubscriptions: () =>
    apiClient.get("/subscriptions/nodes").then((res) => res.data),

  // 根据ID获取节点订阅
  getNodeSubscriptionById: (id: string) =>
    apiClient.get(`/subscriptions/nodes/${id}`).then((res) => res.data),

  // 更新节点订阅
  updateNodeSubscription: (id: string, data: any) =>
    apiClient.put(`/subscriptions/nodes/${id}`, data).then((res) => res.data),

  // 删除节点订阅
  deleteNodeSubscription: (id: string) =>
    apiClient.delete(`/subscriptions/nodes/${id}`).then((res) => res.data),

  // 按币种查询节点订阅
  getNodeSubscriptionsByCryptoType: (cryptoType: string) =>
    apiClient
      .get(`/subscriptions/nodes/crypto/${cryptoType}`)
      .then((res) => res.data),

  // 按风险等级查询节点订阅
  getNodeSubscriptionsByRiskLevel: (riskLevel: string) =>
    apiClient
      .get(`/subscriptions/nodes/risk/${riskLevel}`)
      .then((res) => res.data),

  // 搜索节点订阅
  searchNodeSubscriptions: (keyword: string) =>
    apiClient
      .get(`/subscriptions/nodes/search?keyword=${keyword}`)
      .then((res) => res.data),

  // 切换节点订阅状态
  toggleNodeSubscriptionStatus: (id: string) =>
    apiClient.put(`/subscriptions/nodes/${id}/toggle`).then((res) => res.data),

  // ========== 交易订阅 ==========

  // 创建交易订阅
  createTransactionSubscription: (data: any) =>
    apiClient.post("/subscriptions/transactions", data).then((res) => res.data),

  // 获取所有交易订阅
  getAllTransactionSubscriptions: () =>
    apiClient.get("/subscriptions/transactions").then((res) => res.data),

  // 根据ID获取交易订阅
  getTransactionSubscriptionById: (id: string) =>
    apiClient.get(`/subscriptions/transactions/${id}`).then((res) => res.data),

  // 更新交易订阅
  updateTransactionSubscription: (id: string, data: any) =>
    apiClient
      .put(`/subscriptions/transactions/${id}`, data)
      .then((res) => res.data),

  // 删除交易订阅
  deleteTransactionSubscription: (id: string) =>
    apiClient
      .delete(`/subscriptions/transactions/${id}`)
      .then((res) => res.data),

  // 按币种查询交易订阅
  getTransactionSubscriptionsByCryptoType: (cryptoType: string) =>
    apiClient
      .get(`/subscriptions/transactions/crypto/${cryptoType}`)
      .then((res) => res.data),

  // 按风险等级查询交易订阅
  getTransactionSubscriptionsByRiskLevel: (riskLevel: string) =>
    apiClient
      .get(`/subscriptions/transactions/risk/${riskLevel}`)
      .then((res) => res.data),

  // 搜索交易订阅
  searchTransactionSubscriptions: (keyword: string) =>
    apiClient
      .get(`/subscriptions/transactions/search?keyword=${keyword}`)
      .then((res) => res.data),

  // 切换交易订阅状态
  toggleTransactionSubscriptionStatus: (id: string) =>
    apiClient
      .put(`/subscriptions/transactions/${id}/toggle`)
      .then((res) => res.data),
};

export default { caseApi, subscriptionApi };
