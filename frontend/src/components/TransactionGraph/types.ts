import mockData from "./address_graph_analysis.json";

export interface NodeItem {
  id: string;
  label: string;
  title: string;
  addr: string;
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
}

export interface LinkItem {
  from: string;
  to: string;
  label?: string;
  val: number;
  tx_time: string;
  tx_hash_list: string[];
}

export const sampleData: { nodes: NodeItem[]; links: LinkItem[] } = {
  nodes: mockData.graph_dic.node_list,
  links: mockData.graph_dic.edge_list,
};
