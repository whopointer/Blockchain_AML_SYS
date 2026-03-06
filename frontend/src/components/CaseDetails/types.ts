import { Dayjs } from "dayjs";

export interface GraphSnapshot {
  id: string;
  title: string;
  description: string;
  tags: string[];
  createTime: Dayjs | string;
  nodeCount: number;
  linkCount: number;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  centerAddress?: string;
  fromAddress?: string;
  toAddress?: string;
  hops?: number;
  filterConfig?: {
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: Dayjs | null;
    endDate?: Dayjs | null;
  };
  graphData?: {
    nodes: import("../GraphCommon/types").NodeItem[];
    links: import("../GraphCommon/types").LinkItem[];
  };
}

export interface FilterConfig {
  title: string;
  riskLevel: "LOW" | "MEDIUM" | "HIGH" | "";
  tags: string[];
  dateRange: [Dayjs | null, Dayjs | null];
}

export interface SnapshotTableProps {
  snapshots: GraphSnapshot[];
  filteredSnapshots: GraphSnapshot[];
  loading: boolean;
  filterConfig: FilterConfig;
  allTags: string[];
  onFilterChange: (config: FilterConfig) => void;
  onViewSnapshot: (snapshot: GraphSnapshot) => void;
  onDeleteSnapshot: (snapshot: GraphSnapshot) => void;
  onDownloadSnapshot: (snapshot: GraphSnapshot) => void;
  onClearFilters: () => void;
  editingField?: string | null;
  tempValue?: any;
  startEditing?: (field: string, currentValue: any) => void;
  saveEdit?: (snapshotId: string, field: string) => void;
  cancelEdit?: () => void;
}
