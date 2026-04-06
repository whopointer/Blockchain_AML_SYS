import { GraphSnapshot } from "../../components/CaseDetails/types";

export interface GraphNodeData {
  id: string;
  addr: string;
  label: string;
  title?: string;
  layer: number;
  value?: number;
  pid?: number | string;
  color?: string;
  shape?: string;
  image?: string;
  track?: string;
  expanded?: boolean;
  malicious?: number;
  exg?: number;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphLinkData {
  from: string;
  to: string;
  label?: string;
  val: number;
  tx_time: string;
  tx_hash_list: string[];
}

export interface GraphDataPayload {
  nodes: GraphNodeData[];
  links: GraphLinkData[];
}

export interface CreateSnapshotRequest {
  title: string;
  description?: string;
  tags?: string[];
  nodeCount: number;
  linkCount: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  centerAddress?: string;
  fromAddress?: string;
  toAddress?: string;
  hops?: number;
  filterConfig?: any;
  graphData?: GraphDataPayload;
  dataSource?: "api" | "snapshot" | string;
}

export interface CreateSnapshotResponse {
  success: boolean;
  msg: string;
  data: GraphSnapshot;
}

export interface GetSnapshotsResponse {
  success: boolean;
  msg: string;
  data: GraphSnapshot[];
}

export interface GetSnapshotDetailResponse {
  success: boolean;
  msg: string;
  data: GraphSnapshot;
}

export interface UpdateSnapshotRequest {
  title?: string;
  description?: string;
  tags?: string[];
  riskLevel?: "LOW" | "MEDIUM" | "HIGH";
  filterConfig?: any;
}

export interface UpdateSnapshotResponse {
  success: boolean;
  msg: string;
  data: GraphSnapshot;
}

export interface DeleteSnapshotRequest {
  id: string;
}

export interface DeleteSnapshotResponse {
  success: boolean;
  msg: string;
}
