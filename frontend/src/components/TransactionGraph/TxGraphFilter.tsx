import React, { useMemo, useCallback, useRef, useEffect } from "react";
import {
  Select,
  Row,
  Col,
  InputNumber,
  DatePicker,
  ConfigProvider,
  Slider,
  Space,
} from "antd";
import dayjs, { Dayjs } from "dayjs";
import graphAnalysisData from "./address_graph_analysis.json";

const { Option } = Select;

interface FilterValue {
  txType: "all" | "inflow" | "outflow";
  addrType: "all" | "tagged" | "malicious" | "normal" | "tagged_malicious";
  minAmount?: number;
  maxAmount?: number;
  startDate?: Dayjs | null;
  endDate?: Dayjs | null;
}

interface Props {
  value?: FilterValue;
  onChange?: (v: FilterValue) => void;
}

const TxGraphFilter: React.FC<Props> = ({ value, onChange }) => {
  const lastUpdateRef = useRef<{
    startDate?: dayjs.Dayjs | null;
    endDate?: dayjs.Dayjs | null;
  }>({});

  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  const firstTxTime = useMemo(() => {
    const timeStr = graphAnalysisData.graph_dic.first_tx_datetime;
    return dayjs(timeStr + ":00");
  }, []);

  const latestTxTime = useMemo(() => {
    const timeStr = graphAnalysisData.graph_dic.latest_tx_datetime;
    return dayjs(timeStr + ":00");
  }, []);
  const firstTxTimeMs = useMemo(() => firstTxTime.valueOf(), [firstTxTime]);
  const latestTxTimeMs = useMemo(() => latestTxTime.valueOf(), [latestTxTime]);

  const totalSeconds = useMemo(
    () => (latestTxTimeMs - firstTxTimeMs) / 1000,
    [latestTxTimeMs, firstTxTimeMs]
  );

  const debouncedOnChange = useCallback(
    (newFilter: FilterValue) => {
      if (onChange) {
        if (debounceTimerRef.current) {
          clearTimeout(debounceTimerRef.current);
        }

        debounceTimerRef.current = setTimeout(() => {
          onChange(newFilter);
          debounceTimerRef.current = null;
        }, 5);
      }
    },
    [onChange]
  );

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  const defaultValue = useMemo(
    () => ({
      txType: "all" as const,
      addrType: "all" as const,
      minAmount: undefined,
      maxAmount: undefined,
      startDate: null,
      endDate: null,
    }),
    []
  );

  const sliderValue = useMemo((): [number, number] => {
    const currentValue = value || defaultValue;

    if (!currentValue.startDate || !currentValue.endDate) {
      return [0, totalSeconds];
    }
    const startSeconds =
      (currentValue.startDate.valueOf() - firstTxTimeMs) / 1000;
    const endSeconds = (currentValue.endDate.valueOf() - firstTxTimeMs) / 1000;
    return [
      Math.round(Math.max(0, Math.min(totalSeconds, startSeconds))),
      Math.round(Math.max(0, Math.min(totalSeconds, endSeconds))),
    ];
  }, [value, defaultValue, totalSeconds, firstTxTimeMs]);

  const handleSliderChange = useCallback(
    (values: number | number[]) => {
      if (typeof values === "number") return;

      const [start, end] = values as [number, number];

      const newStartMs = Math.round(firstTxTimeMs + start * 1000);
      const newEndMs = Math.round(firstTxTimeMs + end * 1000);
      const newStartDate = dayjs(newStartMs);
      const newEndDate = dayjs(newEndMs);

      const lastUpdate = lastUpdateRef.current;
      if (
        lastUpdate.startDate?.isSame(newStartDate) &&
        lastUpdate.endDate?.isSame(newEndDate)
      ) {
        return;
      }

      const currentFilterValue = value || defaultValue;

      const newFilter = {
        ...currentFilterValue,
        startDate: newStartDate,
        endDate: newEndDate,
      };

      debouncedOnChange(newFilter);

      lastUpdateRef.current = {
        startDate: newStartDate,
        endDate: newEndDate,
      };
    },
    [debouncedOnChange, firstTxTimeMs, value, defaultValue]
  );

  const handleStartDateChange = useCallback(
    (date: Dayjs | null) => {
      if (!date) {
        const currentFilterValue = value || defaultValue;
        debouncedOnChange({ ...currentFilterValue, startDate: null });
        return;
      }
      const startOfDay = date.startOf("day");
      const currentFilterValue = value || defaultValue;
      debouncedOnChange({ ...currentFilterValue, startDate: startOfDay });
    },
    [value, defaultValue, debouncedOnChange]
  );

  const handleEndDateChange = useCallback(
    (date: Dayjs | null) => {
      if (!date) {
        const currentFilterValue = value || defaultValue;
        debouncedOnChange({ ...currentFilterValue, endDate: null });
        return;
      }
      const endOfDay = date.endOf("day");
      const currentFilterValue = value || defaultValue;
      debouncedOnChange({ ...currentFilterValue, endDate: endOfDay });
    },
    [value, defaultValue, debouncedOnChange]
  );

  return (
    <ConfigProvider
      theme={{
        token: {
          colorBgElevated: "#1a3a52",
          colorPrimary: "#667eea",
          colorText: "#ffffff",
        },
        components: {
          Select: {
            controlItemBgHover: "#3a5f7f",
            controlItemBgActive: "#667eea",
          },
          Tooltip: {
            colorTextLightSolid: "#222",
          },
        },
      }}
    >
      <div
        style={{
          backgroundColor: "#244963",
          border: "1px solid #3a5f7f",
          borderRadius: 8,
          padding: 16,
          height: "100%",
          display: "flex",
          flexDirection: "column",
          color: "#d8e3f0",
        }}
      >
        <Row gutter={12} style={{ marginBottom: 8, alignItems: "center" }}>
          <Col style={{ fontWeight: "bold", color: "#ffffff" }}>数据筛选</Col>
          <Col flex={1}>
            <Select
              value={value?.txType || "all"}
              onChange={(val) => {
                const currentFilterValue = value || defaultValue;
                debouncedOnChange({ ...currentFilterValue, txType: val });
              }}
              style={{ width: "100%" }}
            >
              <Option value="all">全部交易</Option>
              <Option value="inflow">仅收入</Option>
              <Option value="outflow">仅支出</Option>
            </Select>
          </Col>

          <Col flex={1}>
            <Select
              value={value?.addrType || "all"}
              onChange={(val) => {
                const currentFilterValue = value || defaultValue;
                debouncedOnChange({ ...currentFilterValue, addrType: val });
              }}
              style={{ width: "100%" }}
            >
              <Option value="all">全部地址</Option>
              <Option value="tagged">标签地址</Option>
              <Option value="malicious">恶意地址</Option>
              <Option value="normal">普通地址</Option>
              <Option value="tagged_malicious">标签+恶意</Option>
            </Select>
          </Col>
        </Row>

        <Row gutter={12} style={{ alignItems: "center", flexWrap: "nowrap" }}>
          <Col
            style={{
              fontWeight: "bold",
              whiteSpace: "nowrap",
              color: "#ffffff",
            }}
          >
            金额筛选
          </Col>
          <Col flex={1}>
            <InputNumber
              placeholder="最小值"
              value={value?.minAmount}
              onChange={(val) => {
                const currentFilterValue = value || defaultValue;
                debouncedOnChange({
                  ...currentFilterValue,
                  minAmount: val || undefined,
                });
              }}
              style={{ width: "100%" }}
              min={0}
            />
          </Col>
          -
          <Col flex={1}>
            <InputNumber
              placeholder="最大值"
              value={value?.maxAmount}
              onChange={(val) => {
                const currentFilterValue = value || defaultValue;
                debouncedOnChange({
                  ...currentFilterValue,
                  maxAmount: val || undefined,
                });
              }}
              style={{ width: "100%" }}
              min={0}
            />
          </Col>
        </Row>

        <Row
          gutter={12}
          style={{
            marginTop: 12,
            alignItems: "flex-start",
            flexWrap: "nowrap",
          }}
        >
          <Col
            style={{ fontWeight: "bold", whiteSpace: "nowrap", marginTop: 4 }}
          >
            时间筛选
          </Col>
          <Col flex={1}>
            <Space direction="vertical" style={{ width: "100%" }}>
              <Row gutter={8} style={{ fontSize: 12, color: "#9bb3c8" }}>
                <Col style={{ flex: 1 }}>
                  <DatePicker
                    placeholder="起始日期"
                    value={value?.startDate || null}
                    onChange={handleStartDateChange}
                    format="YYYY-MM-DD"
                    style={{ width: "100%" }}
                  />
                </Col>
                <Col style={{ flex: 1 }}>
                  <DatePicker
                    placeholder="结束日期"
                    value={value?.endDate || null}
                    onChange={handleEndDateChange}
                    format="YYYY-MM-DD"
                    style={{ width: "100%" }}
                  />
                </Col>
              </Row>
              <Slider
                range
                min={0}
                max={totalSeconds}
                value={sliderValue}
                onChange={handleSliderChange}
                marks={{
                  0: (
                    <span style={{ fontSize: 10, color: "#9bb3c8" }}>最早</span>
                  ),
                  [totalSeconds]: (
                    <span style={{ fontSize: 10, color: "#9bb3c8" }}>最近</span>
                  ),
                }}
                tooltip={{
                  formatter: (value) => {
                    if (value === undefined) return "";
                    return dayjs(firstTxTimeMs + value * 1000).format(
                      "YYYY-MM-DD HH:mm:ss"
                    );
                  },
                }}
                trackStyle={[{ backgroundColor: "#667eea" }]}
                railStyle={{ backgroundColor: "#3a5f7f" }}
                handleStyle={[
                  {
                    backgroundColor: "#667eea",
                    borderColor: "#667eea",
                    boxShadow: "0 0 0 3px rgba(102, 126, 234, 0.2)",
                  },
                  {
                    backgroundColor: "#667eea",
                    borderColor: "#667eea",
                    boxShadow: "0 0 0 3px rgba(102, 126, 234, 0.2)",
                  },
                ]}
              />
            </Space>
          </Col>
        </Row>
      </div>
    </ConfigProvider>
  );
};

export default TxGraphFilter;
