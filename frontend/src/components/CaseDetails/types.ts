import { Dayjs } from "dayjs";

export interface CaseComment {
  id: string;
  author: string;
  content: string;
  createdAt: string;
}

export interface GraphSnapshot {
  id: string;
  title: string;
  description: string;
  tags: string[];
  createTime: Dayjs | string;
  nodeCount: number;
  linkCount: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  archived?: boolean;
  comments?: CaseComment[];
  centerAddress?: string;
  fromAddress?: string;
  toAddress?: string;
  hops?: number;
  chain?: string;
  filterConfig?: {
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: Dayjs | null;
    endDate?: Dayjs | null;
  };
  transformConfig?: {
    x: number;
    y: number;
    k: number;
  };
  graphData?: {
    nodes: import("../GraphCommon/types").NodeItem[];
    links: import("../GraphCommon/types").LinkItem[];
  };
  dataSource?: "api" | "snapshot";
}

export interface FilterConfig {
  title: string;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "";
  tags: string[];
  dateRange: [Dayjs | null, Dayjs | null];
}

export type CaseStatusFilter = "ALL" | "ACTIVE" | "ARCHIVED";

// 案件管理相关类型
export interface Case {
  id: string;
  title: string;
  description: string;
  status: "NEW" | "IN_PROGRESS" | "ARCHIVED" | "CLOSED";
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  tags: string[];
  createTime: Dayjs | string;
  updateTime: Dayjs | string;
  assignedTo?: string;
  priority: "LOW" | "MEDIUM" | "HIGH" | "URGENT";
  relatedSnapshots?: string[];
  comments?: CaseComment[];
}

export interface CaseFilterConfig {
  keyword: string;
  status: "ALL" | "NEW" | "IN_PROGRESS" | "ARCHIVED" | "CLOSED";
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "";
  priority: "LOW" | "MEDIUM" | "HIGH" | "URGENT" | "";
  tags: string[];
  dateRange: [Dayjs | null, Dayjs | null];
}

// 订阅相关类型
export interface SubscribedNode {
  id: string;
  address: string;
  label?: string;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  tags: string[];
  remark: string;
  subscribedAt: Dayjs | string;
  lastActivity?: Dayjs | string;
  alertEnabled: boolean;
  relatedCases?: string[];
}

export interface SubscribedTransaction {
  id: string;
  txHash: string;
  fromAddress: string;
  toAddress: string;
  amount: string;
  token: string;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  tags: string[];
  remark: string;
  subscribedAt: Dayjs | string;
  txTime?: Dayjs | string;
  alertEnabled: boolean;
  relatedCases?: string[];
}

export interface SubscriptionFilter {
  keyword: string;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "";
  tags: string[];
  alertOnly: boolean;
}

// 侧边栏菜单项
export type MenuKey = "case-management" | "graph-snapshot" | "subscription";

export interface SidebarProps {
  activeKey: MenuKey;
  onMenuSelect: (key: MenuKey) => void;
  collapsed?: boolean;
}

// 图谱快照表格 Props
export interface SnapshotTableProps {
  filteredSnapshots: GraphSnapshot[];
  snapshots: GraphSnapshot[];
  loading: boolean;
  filterConfig: FilterConfig;
  statusFilter: CaseStatusFilter;
  allTags: string[];
  onFilterChange: (config: FilterConfig) => void;
  onStatusFilterChange?: (status: CaseStatusFilter) => void;
  onViewSnapshot: (snapshot: GraphSnapshot) => void;
  onDownloadSnapshot?: (snapshot: GraphSnapshot) => void;
  onDeleteSnapshot: (snapshot: GraphSnapshot) => void;
  onToggleArchive: (snapshot: GraphSnapshot) => void;
  onClearFilters: () => void;
  onExportPDF?: (snapshot: GraphSnapshot) => void;
  editingField?: string | null;
  tempValue?: any;
  startEditing?: (field: string, currentValue: any) => void;
  saveEdit?: (snapshotId: string, field: string) => void;
  cancelEdit?: () => void;
}
