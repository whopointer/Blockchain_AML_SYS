import React from "react";
import { useNavigate } from "react-router-dom";
import { Select, Input, Button } from "antd";
import { SearchOutlined } from "@ant-design/icons";

interface SearchBarProps {
  defaultCrypto?: string;
  defaultAddress?: string;
  defaultHops?: number;
}

const SearchBar: React.FC<SearchBarProps> = ({
  defaultCrypto = "BNB",
  defaultAddress = "",
  defaultHops = 1,
}) => {
  const navigate = useNavigate();
  const [crypto, setCrypto] = React.useState<string>(defaultCrypto);
  const [address, setAddress] = React.useState<string>(defaultAddress);
  const [hops, setHops] = React.useState<number>(defaultHops);

  const handleSearch = () => {
    if (!address) return;
    navigate(
      `/transaction-graph/${crypto.toLowerCase()}/${address}?hops=${hops}`,
    );
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        margin: "0 auto 24px auto",
        gap: 0,
        width: "80%",
      }}
    >
      <Select
        value={crypto}
        onChange={setCrypto}
        style={{
          width: 120,
          borderTopRightRadius: 0,
          borderBottomRightRadius: 0,
        }}
        options={[
          { value: "BNB", label: "BNB" },
          { value: "ETH", label: "ETH" },
        ]}
        size="large"
      />
      <Input
        placeholder="输入钱包地址 / ENS"
        value={address}
        onChange={(e) => setAddress(e.target.value)}
        onKeyPress={handleKeyPress}
        style={{
          flex: 1,
borderTopLeftRadius: 0,
          borderBottomLeftRadius: 0,
        }}
        size="large"
      />
      <Select
        value={hops}
        onChange={setHops}
        style={{
          width: 100,
          marginLeft: "8px",
        }}
        options={[
          { value: 1, label: "1跳" },
          { value: 2, label: "2跳" },
          { value: 3, label: "3跳" },
          { value: 4, label: "4跳" },
          { value: 5, label: "5跳" },
        ]}
        placeholder="跳数"
        size="large"
      />

      <Button
        type="primary"
        icon={<SearchOutlined />}
        onClick={handleSearch}
        style={{
          marginLeft: "18px",
        }}
        size="large"
      >
        查询
      </Button>
    </div>
  );
};

export default SearchBar;
