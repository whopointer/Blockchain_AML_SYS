import { GraphSnapshot } from "../../components/CaseDetails/types";

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
}

export interface CreateSnapshotResponse {
  success: boolean;
  msg: string;
  data: GraphSnapshot;
}

export interface GetSnapshotResponse {
  success: boolean;
  msg: string;
  data: GraphSnapshot[];
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
