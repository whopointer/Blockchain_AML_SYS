import axios from "axios";
import {
  CreateSnapshotRequest,
  CreateSnapshotResponse,
  GetSnapshotsResponse,
  GetSnapshotDetailResponse,
  DeleteSnapshotResponse,
  UpdateSnapshotRequest,
  UpdateSnapshotResponse,
} from "./types";

const GRAPH_SNAPSHOT_API_BASE_URL = "http://localhost:8081/api";

const graphSnapshotApiClient = axios.create({
  baseURL: GRAPH_SNAPSHOT_API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

export const graphSnapshotApi = {
  // 创建图谱快照
  createSnapshot: (
    snapshot: CreateSnapshotRequest,
  ): Promise<CreateSnapshotResponse> =>
    graphSnapshotApiClient
      .post("/neo4j/snapshot", snapshot)
      .then((response) => response.data),

  // 获取所有图谱快照（不包含 graphData）
  getSnapshots: (): Promise<GetSnapshotsResponse> =>
    graphSnapshotApiClient
      .get("/neo4j/snapshots")
      .then((response) => response.data),

  // 获取单个图谱快照详情（包含 graphData）
  getSnapshotDetail: (id: string): Promise<GetSnapshotDetailResponse> =>
    graphSnapshotApiClient
      .get(`/neo4j/snapshot/${id}`)
      .then((response) => response.data),

  // 修改图谱快照信息
  updateSnapshot: (
    id: string,
    snapshot: UpdateSnapshotRequest,
  ): Promise<UpdateSnapshotResponse> =>
    graphSnapshotApiClient
      .put(`/neo4j/snapshot/${id}`, snapshot)
      .then((response) => response.data),

  // 删除图谱快照
  deleteSnapshot: (id: string): Promise<DeleteSnapshotResponse> =>
    graphSnapshotApiClient
      .delete(`/neo4j/snapshot/${id}`)
      .then((response) => response.data),
};

export default graphSnapshotApi;
