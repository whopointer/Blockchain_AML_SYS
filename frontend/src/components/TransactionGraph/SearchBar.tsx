import React from "react";
import { useNavigate } from "react-router-dom";
import { Select, Input, Button, message } from "antd";
import { SearchOutlined } from "@ant-design/icons";

interface SearchBarProps {
  defaultCrypto?: string;
  defaultAddress?: string;
  defaultHops?: number;
}

// 以太坊地址校验
const isValidEthAddress = (address: string): boolean => {
  // 以太坊地址：0x 开头，后跟 40 位十六进制字符
  return /^0x[a-fA-F0-9]{40}$/.test(address);
};

// 比特币地址校验（支持 P2PKH, P2SH, Bech32）
const isValidBtcAddress = (address: string): boolean => {
  // P2PKH: 1 开头，26-35 位
  // P2SH: 3 开头，26-35 位
  // Bech32: bc1 开头
  return (
    /^1[a-zA-Z0-9]{25,34}$/.test(address) ||
    /^3[a-zA-Z0-9]{25,34}$/.test(address) ||
    /^bc1[a-zA-Z0-9]{6,87}$/i.test(address)
  );
};

// ENS 域名校验
const isValidEns = (address: string): boolean => {
  // ENS: 以 .eth 结尾
  return /^[a-zA-Z0-9-]+\.eth$/i.test(address);
};

// 综合地址校验
const validateAddress = (crypto: string, address: string): string | null => {
  if (!address || address.trim() === "") {
    return "请输入钱包地址";
  }

  const trimmedAddress = address.trim();

  if (crypto === "ETH") {
    // ETH 支持地址和 ENS
    if (isValidEns(trimmedAddress)) {
      return null; // ENS 域名有效
    }
    if (!isValidEthAddress(trimmedAddress)) {
      return "请输入有效的以太坊地址（0x 开头，40位十六进制）或 ENS 域名";
    }
  } else if (crypto === "BTC") {
    if (!isValidBtcAddress(trimmedAddress)) {
      return "请输入有效的比特币地址（以 1、3 或 bc1 开头）";
    }
  }

  return null;
};

const SearchBar: React.FC<SearchBarProps> = ({
  defaultCrypto = "BTC",
  defaultAddress = "",
  defaultHops = 1,
}) => {
  const navigate = useNavigate();
  const [crypto, setCrypto] = React.useState<string>(defaultCrypto);
  const [address, setAddress] = React.useState<string>(defaultAddress);
  const [hops, setHops] = React.useState<number>(defaultHops);
  const [error, setError] = React.useState<string | null>(null);

  const handleSearch = () => {
    const trimmedAddress = address.trim();
    const validationError = validateAddress(crypto, trimmedAddress);

    if (validationError) {
      setError(validationError);
      message.error(validationError);
      return;
    }

    setError(null);
    navigate(
      `/transaction-graph/${crypto.toLowerCase()}/${trimmedAddress}?hops=${hops}`,
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
        onChange={(value) => {
          setCrypto(value);
          setError(null);
        }}
        style={{
          width: 120,
          borderTopRightRadius: 0,
          borderBottomRightRadius: 0,
        }}
        options={[
          { value: "BTC", label: "BTC" },
          { value: "ETH", label: "ETH" },
        ]}
        size="large"
      />
      <Input
        placeholder={
          crypto === "ETH" ? "输入以太坊地址 / ENS" : "输入比特币地址"
        }
        value={address}
        onChange={(e) => {
          setAddress(e.target.value);
          setError(null);
        }}
        onKeyPress={handleKeyPress}
        status={error ? "error" : undefined}
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
