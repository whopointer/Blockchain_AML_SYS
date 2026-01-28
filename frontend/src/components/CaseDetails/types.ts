import { Dayjs } from "dayjs";
import { NodeItem, LinkItem } from "../TransactionGraph/types";

export interface GraphSnapshot {
  id: string;
  title: string;
  description: string;
  tags: string[];
  createTime: Dayjs | string;
  mainAddress: string;
  nodeCount: number;
  linkCount: number;
  riskLevel: "low" | "medium" | "high";
  graphData?: {
    nodes: NodeItem[];
    links: LinkItem[];
  };
  filterConfig?: {
    txType: "all" | "inflow" | "outflow";
    addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
    minAmount?: number;
    maxAmount?: number;
    startDate?: Dayjs | null;
    endDate?: Dayjs | null;
  };
}

export interface FilterConfig {
  title: string;
  riskLevel: string;
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
