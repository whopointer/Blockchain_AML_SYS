import React from "react";
import SearchBar from "./SearchBar";

interface TransactionGraphSearchProps {
  currencySymbol: string;
}

const TransactionGraphSearch: React.FC<TransactionGraphSearchProps> = ({
  currencySymbol,
}) => {
  return (
    <div className="dashboard" style={{ margin: "32px 0" }}>
      <div className="text-center mb-8" style={{ padding: "48px 0" }}>
        <h2 className="mb-3">交易图谱查询</h2>
        <p className="text-secondary mb-6">
          通过地址和加密货币类型查询交易关系图谱
        </p>
      </div>

      <div style={{ maxWidth: 800, margin: "0 auto" }}>
        <SearchBar
          defaultCrypto={currencySymbol}
          defaultAddress=""
          defaultHops={1}
        />
      </div>
    </div>
  );
};

export default TransactionGraphSearch;
