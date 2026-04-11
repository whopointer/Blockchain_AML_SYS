import React from "react";
import SearchBar from "./SearchBar";

interface PathTrackingSearchProps {
  form: any;
  onFinish: (values: any) => void;
  loading: boolean;
  routeCrypto?: string | null;
  urlFromAddress?: string | null;
  urlToAddress?: string | null;
}

const PathTrackingSearch: React.FC<PathTrackingSearchProps> = ({
  form,
  onFinish,
  loading,
  routeCrypto,
  urlFromAddress,
  urlToAddress,
}) => {
  return (
    <div className="dashboard" style={{ margin: "32px 0" }}>
      <div className="text-center mb-8" style={{ padding: "48px 0" }}>
        <h2 className="mb-3">交易路径查询</h2>
        <p className="text-secondary mb-6">
          通过起始地址和目标地址查询交易路径关系
        </p>
        <div style={{ maxWidth: 800, padding: "0 16px", margin: "0 auto" }}>
          <SearchBar
            defaultCrypto={routeCrypto || "eth"}
            defaultFromAddress={urlFromAddress || ""}
            defaultToAddress={urlToAddress || ""}
          />
        </div>
      </div>
    </div>
  );
};

export default PathTrackingSearch;
