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
  fx?: number | null;
  fy?: number | null;
  type?: "address" | "transaction";
  txHash?: string;
  blockHeight?: number;
  time?: string;
}

export interface LinkItem {
  from: string;
  to: string;
  label?: string;
  val: number;
  tx_time: string;
  tx_hash_list: string[];
}
