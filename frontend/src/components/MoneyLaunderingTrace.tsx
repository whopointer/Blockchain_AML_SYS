import React, { useEffect, useState } from "react";
import {
  Form,
  Button,
  Alert,
  Spinner,
  Card,
  Badge,
  Row,
  Col,
  Table,
  ListGroup,
  OverlayTrigger,
  Tooltip,
} from "react-bootstrap";
import {
  api,
  TraceByTimeRequest,
  TraceTaskResult,
  TraceTaskStatus,
  TRACER_ORIGIN,
} from "../services/api";

const TRACE_FILE_BASE_URL = TRACER_ORIGIN;
const TRACE_PARAM_DEFAULTS = {
  out_degree_limit: 1000,
  depth: 10,
  activate_threshold: 100.0,
  age_limit: 10,
  label_limit: 3000,
};

const parseAddressList = (raw: string): string[] =>
  raw
    .split(/[\n,]/)
    .map((v) => v.trim())
    .filter(Boolean);

const fmtNumber = (v?: number): string =>
  typeof v === "number"
    ? v.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : "-";

const shortAddr = (v: string): string =>
  v.length > 18 ? `${v.slice(0, 10)}...${v.slice(-6)}` : v;

const getTaskVariant = (status?: string) => {
  if (status === "completed") return "success";
  if (status === "running") return "warning";
  if (status === "failed" || status === "error") return "danger";
  return "secondary";
};

const buildApiUrl = (path: string): string => {
  if (!path) return "#";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${TRACE_FILE_BASE_URL}${path}`;
};

const MoneyLaunderingTrace: React.FC = () => {
  const [formData, setFormData] = useState({
    start_time: "2023-06-04",
    end_time: "2023-06-05",
    token: "ETH" as "USDC" | "USDT" | "DAI" | "WETH" | "ETH",
    src: "",
    allowed: "",
    forbidden: "",
    out_degree_limit: "",
    depth: "",
    activate_threshold: "",
    age_limit: "",
    label_limit: "",
  });

  const [error, setError] = useState<string>("");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [task, setTask] = useState<TraceTaskStatus | null>(null);
  const [polling, setPolling] = useState<boolean>(false);
  const [vizLoading, setVizLoading] = useState<boolean>(false);
  const [vizError, setVizError] = useState<string>("");
  const [svgUrl, setSvgUrl] = useState<string>("");
  const [svgContent, setSvgContent] = useState<string>("");

  const guessFilenameFromUrl = (url: string, fallback: string) => {
    try {
      const u = url.startsWith("http")
        ? new URL(url)
        : new URL(url, window.location.origin);
      const last = u.pathname.split("/").filter(Boolean).pop();
      if (last && last.includes(".")) return decodeURIComponent(last);
    } catch {
      // ignore
    }
    return fallback;
  };

  const downloadFromUrl = async (urlOrPath: string, filename: string) => {
    const url = buildApiUrl(urlOrPath);
    if (!url || url === "#") return;
    try {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`下载失败（HTTP ${res.status}）`);
      }
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = guessFilenameFromUrl(url, filename);
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (e: any) {
      const msg = e?.message || "下载失败";
      setError(msg);
    }
  };
  const [vizConfig, setVizConfig] = useState({
    simplify: false,
    min_value: "100",
    max_nodes: "500",
    layout: "fdp" as "fdp" | "dot" | "neato" | "sfdp" | "twopi" | "circo",
  });

  useEffect(() => {
    if (!task?.request_id) return undefined;
    if (
      task.status === "completed" ||
      task.status === "failed" ||
      task.status === "error"
    )
      return undefined;

    setPolling(true);
    const timer = window.setInterval(async () => {
      try {
        const status = await api.getTraceTaskStatus(task.request_id);
        setTask(status);
        if (
          status.status === "completed" ||
          status.status === "failed" ||
          status.status === "error"
        ) {
          window.clearInterval(timer);
          setPolling(false);
        }
      } catch (e: any) {
        window.clearInterval(timer);
        setPolling(false);
        const msg = e.response?.data?.error || e.message || "任务状态查询失败";
        setError(msg);
      }
    }, 3000);

    return () => {
      window.clearInterval(timer);
    };
  }, [task?.request_id, task?.status]);

  const onChange = (key: keyof typeof formData, value: string | number) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const toOptionalInt = (v: string): number | undefined => {
    if (!v.trim()) return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? Math.trunc(n) : undefined;
  };

  const toOptionalFloat = (v: string): number | undefined => {
    if (!v.trim()) return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setTask(null);
    setSvgUrl("");
    setSvgContent("");
    setVizError("");
    setSubmitting(true);

    try {
      const src = parseAddressList(formData.src);
      if (src.length === 0) {
        setError("src 源地址列表不能为空");
        return;
      }
      const startT = formData.start_time.trim();
      const endT = formData.end_time.trim();
      if (!startT || !endT) {
        setError(
          "请填写起始时间与终止时间（支持 YYYY-MM-DD，与文档一致按 UTC 日界线）",
        );
        return;
      }
      if (startT > endT) {
        setError("起始时间不能晚于终止时间");
        return;
      }

      const allowed = parseAddressList(formData.allowed);
      const forbidden = parseAddressList(formData.forbidden);
      const cleanedRequest: TraceByTimeRequest = {
        start_time: startT,
        end_time: endT,
        token: formData.token,
        src,
      };
      if (allowed.length > 0) cleanedRequest.allowed = allowed;
      if (forbidden.length > 0) cleanedRequest.forbidden = forbidden;
      const outDegreeLimit = toOptionalInt(formData.out_degree_limit);
      if (outDegreeLimit !== undefined)
        cleanedRequest.out_degree_limit = outDegreeLimit;
      const depth = toOptionalInt(formData.depth);
      if (depth !== undefined) cleanedRequest.depth = depth;
      const activateThreshold = toOptionalFloat(formData.activate_threshold);
      if (activateThreshold !== undefined)
        cleanedRequest.activate_threshold = activateThreshold;
      const ageLimit = toOptionalInt(formData.age_limit);
      if (ageLimit !== undefined) cleanedRequest.age_limit = ageLimit;
      const labelLimit = toOptionalInt(formData.label_limit);
      if (labelLimit !== undefined) cleanedRequest.label_limit = labelLimit;

      const created = await api.createTraceTask(cleanedRequest);
      setTask(created);
    } catch (err: any) {
      const msg =
        err.response?.data?.error || err.message || "提交追踪任务失败";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  const handleGenerateVisualization = async () => {
    if (!task?.request_id) return;
    setVizError("");
    setVizLoading(true);
    try {
      const res = await api.generateTraceVisualization({
        request_id: task.request_id,
        simplify: vizConfig.simplify,
        min_value: Number(vizConfig.min_value || "100"),
        max_nodes: Number(vizConfig.max_nodes || "500"),
        layout: vizConfig.layout,
      });
      if (res.status !== "success" || !res.svg_url) {
        setVizError(res.error || "SVG 生成失败");
        return;
      }
      const svgPath = res.svg_url.startsWith("http")
        ? new URL(res.svg_url).pathname + (new URL(res.svg_url).search || "")
        : res.svg_url;
      const content = await api.getTraceSvgContent(svgPath);
      setSvgUrl(buildApiUrl(res.svg_url));
      setSvgContent(content);
    } catch (e: any) {
      const msg = e.response?.data?.error || e.message || "调用可视化接口失败";
      setVizError(msg);
    } finally {
      setVizLoading(false);
    }
  };

  const renderParamLabel = (
    name: string,
    description: string,
    required = false,
  ) => (
    <OverlayTrigger
      placement="top"
      overlay={<Tooltip id={`tip-${name}`}>{description}</Tooltip>}
    >
      <Form.Label style={{ cursor: "help" }}>
        {name}
        {required ? " *" : ""}
        <small className="text-muted ms-1">ⓘ</small>
      </Form.Label>
    </OverlayTrigger>
  );

  const applyDefaultConfig = () => {
    setFormData((prev) => ({
      ...prev,
      out_degree_limit: String(TRACE_PARAM_DEFAULTS.out_degree_limit),
      depth: String(TRACE_PARAM_DEFAULTS.depth),
      activate_threshold: String(TRACE_PARAM_DEFAULTS.activate_threshold),
      age_limit: String(TRACE_PARAM_DEFAULTS.age_limit),
      label_limit: String(TRACE_PARAM_DEFAULTS.label_limit),
    }));
  };

  const renderResult = (result?: TraceTaskResult) => {
    if (!result) return null;

    const nodes = result.nodes || [];
    const edges = result.edges || [];
    const files = result.files || {};

    return (
      <>
        <Row className="g-3 mb-3">
          <Col md={6}>
            <Card>
              <Card.Header>Meta</Card.Header>
              <Card.Body>
                <div>
                  <strong>request_id:</strong> {result.request_id}
                </div>
                <div>
                  <strong>status:</strong> {result.status || "-"}
                </div>
                <div>
                  <strong>区块范围:</strong>{" "}
                  {result.meta?.start_block_id ?? "-"} -{" "}
                  {result.meta?.end_block_id ?? "-"}
                </div>
                <div>
                  <strong>token:</strong> {result.meta?.token || "-"}
                </div>
                <div>
                  <strong>token_address:</strong>{" "}
                  {result.meta?.token_address || "-"}
                </div>
                <div>
                  <strong>activate_threshold_usd:</strong>{" "}
                  {fmtNumber(result.meta?.activate_threshold_usd)}
                </div>
                <div>
                  <strong>age_limit:</strong> {result.meta?.age_limit ?? "-"}
                </div>
                <div>
                  <strong>execution_time_ms:</strong>{" "}
                  {result.meta?.execution_time_ms ?? "-"}
                </div>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6}>
            <Card>
              <Card.Header>Summary</Card.Header>
              <Card.Body>
                <div>
                  <strong>source_address_count:</strong>{" "}
                  {result.summary?.source_address_count ?? "-"}
                </div>
                <div>
                  <strong>node_count:</strong>{" "}
                  {result.summary?.node_count ?? "-"}
                </div>
                <div>
                  <strong>edge_count:</strong>{" "}
                  {result.summary?.edge_count ?? "-"}
                </div>
                <div>
                  <strong>total_flow_usd:</strong>{" "}
                  {fmtNumber(result.summary?.total_flow_usd)}
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        <Card className="mb-3">
          <Card.Header>图形可视化（SVG）</Card.Header>
          <Card.Body>
            <Row className="g-2 align-items-end mb-3">
              <Col md={3}>
                <Form.Check
                  type="switch"
                  id="viz-simplify"
                  label="simplify"
                  checked={vizConfig.simplify}
                  onChange={(e) =>
                    setVizConfig((prev) => ({
                      ...prev,
                      simplify: e.target.checked,
                    }))
                  }
                />
              </Col>
              <Col md={3}>
                <Form.Group>
                  <Form.Label>min_value</Form.Label>
                  <Form.Control
                    type="number"
                    step="0.01"
                    value={vizConfig.min_value}
                    onChange={(e) =>
                      setVizConfig((prev) => ({
                        ...prev,
                        min_value: e.target.value,
                      }))
                    }
                  />
                </Form.Group>
              </Col>
              <Col md={3}>
                <Form.Group>
                  <Form.Label>max_nodes</Form.Label>
                  <Form.Control
                    type="number"
                    value={vizConfig.max_nodes}
                    onChange={(e) =>
                      setVizConfig((prev) => ({
                        ...prev,
                        max_nodes: e.target.value,
                      }))
                    }
                  />
                </Form.Group>
              </Col>
              <Col md={3}>
                <Form.Group>
                  <Form.Label>layout</Form.Label>
                  <Form.Select
                    value={vizConfig.layout}
                    onChange={(e) =>
                      setVizConfig((prev) => ({
                        ...prev,
                        layout: e.target.value as
                          | "fdp"
                          | "dot"
                          | "neato"
                          | "sfdp"
                          | "twopi"
                          | "circo",
                      }))
                    }
                  >
                    <option value="fdp">fdp</option>
                    <option value="dot">dot</option>
                    <option value="neato">neato</option>
                    <option value="sfdp">sfdp</option>
                    <option value="twopi">twopi</option>
                    <option value="circo">circo</option>
                  </Form.Select>
                </Form.Group>
              </Col>
            </Row>

            <div className="d-flex gap-2 mb-3">
              <Button
                onClick={handleGenerateVisualization}
                disabled={vizLoading}
              >
                {vizLoading ? (
                  <>
                    <Spinner
                      as="span"
                      animation="border"
                      size="sm"
                      className="me-2"
                    />
                    生成中...
                  </>
                ) : (
                  "生成图形"
                )}
              </Button>
              {svgUrl && (
                <Button
                  variant="outline-secondary"
                  type="button"
                  onClick={() => downloadFromUrl(svgUrl, "trace_flow.svg")}
                >
                  下载SVG文件
                </Button>
              )}
            </div>

            {vizError && (
              <Alert variant="danger" className="mb-3">
                {vizError}
              </Alert>
            )}

            {svgUrl ? (
              <div
                style={{
                  border: "1px solid #dee2e6",
                  borderRadius: 8,
                  overflow: "hidden",
                }}
              >
                {svgContent ? (
                  <div
                    style={{
                      width: "100%",
                      minHeight: 520,
                      overflow: "auto",
                      background: "#fff",
                    }}
                    // SVG 由同一后端返回，仅用于可视化展示
                    dangerouslySetInnerHTML={{ __html: svgContent }}
                  />
                ) : (
                  <object
                    data={svgUrl}
                    type="image/svg+xml"
                    style={{ width: "100%", minHeight: 520 }}
                  >
                    <div className="p-3">
                      浏览器无法内嵌 SVG，请点击“新窗口打开”查看。
                    </div>
                  </object>
                )}
              </div>
            ) : (
              <div className="text-muted">
                点击“生成图形”后，在此展示资金流向 SVG。
              </div>
            )}
          </Card.Body>
        </Card>

        <Card className="mb-3">
          <Card.Header className="d-flex justify-content-between align-items-center">
            <span>Nodes（展示前 20 条）</span>
            <Badge bg="secondary">{nodes.length}</Badge>
          </Card.Header>
          <Card.Body style={{ maxHeight: 320, overflowY: "auto" }}>
            {nodes.length === 0 ? (
              <div className="text-muted">无节点数据</div>
            ) : (
              <Table responsive hover size="sm">
                <thead>
                  <tr>
                    <th>address</th>
                    <th>in_flow_usd</th>
                    <th>out_flow_usd</th>
                    <th>net_flow_usd</th>
                    <th>is_source</th>
                    <th>is_exchange</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.slice(0, 20).map((node, idx) => (
                    <tr key={`${node.address}-${idx}`}>
                      <td title={node.address}>{shortAddr(node.address)}</td>
                      <td>{fmtNumber(node.in_flow_usd)}</td>
                      <td>{fmtNumber(node.out_flow_usd)}</td>
                      <td>{fmtNumber(node.net_flow_usd)}</td>
                      <td>{node.is_source ? "Y" : "N"}</td>
                      <td>{node.is_exchange ? "Y" : "N"}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Card.Body>
        </Card>

        <Card className="mb-3">
          <Card.Header className="d-flex justify-content-between align-items-center">
            <span>Edges（展示前 20 条）</span>
            <Badge bg="secondary">{edges.length}</Badge>
          </Card.Header>
          <Card.Body style={{ maxHeight: 320, overflowY: "auto" }}>
            {edges.length === 0 ? (
              <div className="text-muted">无边数据</div>
            ) : (
              <Table responsive hover size="sm">
                <thead>
                  <tr>
                    <th>from</th>
                    <th>to</th>
                    <th>tx_hash</th>
                    <th>block</th>
                    <th>amount_usd</th>
                    <th>age</th>
                  </tr>
                </thead>
                <tbody>
                  {edges.slice(0, 20).map((edge, idx) => (
                    <tr key={`${edge.tx_hash || idx}-${idx}`}>
                      <td title={edge.from}>{shortAddr(edge.from)}</td>
                      <td title={edge.to}>{shortAddr(edge.to)}</td>
                      <td title={edge.tx_hash}>
                        {edge.tx_hash ? shortAddr(edge.tx_hash) : "-"}
                      </td>
                      <td>{edge.block_id ?? "-"}</td>
                      <td>{fmtNumber(edge.amount_usd)}</td>
                      <td>{edge.age ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Card.Body>
        </Card>

        <Card>
          <Card.Header>结果文件下载</Card.Header>
          <ListGroup variant="flush">
            {Object.entries(files).length === 0 ? (
              <ListGroup.Item className="text-muted">
                无文件下载信息
              </ListGroup.Item>
            ) : (
              Object.entries(files).map(([k, v]) => (
                <ListGroup.Item
                  key={k}
                  className="d-flex justify-content-between align-items-center"
                >
                  <code>{k}</code>
                  <Button
                    size="sm"
                    variant="outline-primary"
                    type="button"
                    onClick={() => downloadFromUrl(v, k)}
                  >
                    下载
                  </Button>
                </ListGroup.Item>
              ))
            )}
          </ListGroup>
        </Card>
      </>
    );
  };

  return (
    <div className="money-laundering-trace" style={{ marginTop: "20px" }}>
      <Card>
        <Card.Body>
          <div className="text-center mb-4">
            <h3>💰 洗钱追踪分析</h3>
            <p className="text-secondary">通过时间范围和地址参数追踪资金流向</p>
          </div>

          <Row>
            <Col xs={12} className="mb-4">
              <Card>
                <Card.Header className="d-flex justify-content-between align-items-center">
                  <div>
                    <h5 className="mb-0">追踪配置（Tracer）</h5>
                    <small className="text-muted">
                      填写参数后提交，支持一键回填默认配置
                    </small>
                  </div>
                  <Button
                    variant="outline-secondary"
                    size="sm"
                    type="button"
                    onClick={applyDefaultConfig}
                    disabled={submitting || polling}
                  >
                    默认配置
                  </Button>
                </Card.Header>
                <Card.Body>
                  <Form onSubmit={handleSubmit}>
                    <Row className="g-3">
                      <Col md={6}>
                        <Form.Group>
                          {renderParamLabel(
                            "start_time",
                            "起始时间（自然日）。支持 YYYY-MM-DD 或 RFC3339；后端按 UTC 取日界线，与 README「按时间范围追踪」一致",
                            true,
                          )}
                          <Form.Control
                            type="date"
                            value={formData.start_time}
                            onChange={(e) =>
                              onChange("start_time", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group>
                          {renderParamLabel(
                            "end_time",
                            "终止时间（自然日，含当日）。与 start_time 格式一致",
                            true,
                          )}
                          <Form.Control
                            type="date"
                            value={formData.end_time}
                            onChange={(e) =>
                              onChange("end_time", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group>
                          {renderParamLabel(
                            "token",
                            "代币类型，支持 USDT 或 ETH",
                            true,
                          )}
                          <Form.Select
                            value={formData.token}
                            onChange={(e) => onChange("token", e.target.value)}
                            disabled={submitting || polling}
                          >
                            <option value="USDC">USDC</option>
                            <option value="USDT">USDT</option>
                            <option value="DAI">DAI</option>
                            <option value="WETH">WETH</option>
                            <option value="ETH">ETH</option>
                          </Form.Select>
                        </Form.Group>
                      </Col>
                      <Col md={6}>
                        <Form.Group>
                          {renderParamLabel(
                            "out_degree_limit",
                            "节点出度限制，用于防止在高度连接节点（如交易所）上花费过多时间",
                          )}
                          <Form.Control
                            type="number"
                            value={formData.out_degree_limit}
                            onChange={(e) =>
                              onChange("out_degree_limit", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={12}>
                        <Form.Group>
                          {renderParamLabel(
                            "src（源地址列表）",
                            "源地址列表",
                            true,
                          )}
                          <Form.Control
                            as="textarea"
                            rows={3}
                            placeholder="每行一个地址，或逗号分隔"
                            value={formData.src}
                            onChange={(e) => onChange("src", e.target.value)}
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={12}>
                        <Form.Group>
                          {renderParamLabel(
                            "allowed（可选）",
                            "允许通过的地址列表",
                          )}
                          <Form.Control
                            as="textarea"
                            rows={2}
                            placeholder="每行一个地址，或逗号分隔"
                            value={formData.allowed}
                            onChange={(e) =>
                              onChange("allowed", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={12}>
                        <Form.Group>
                          {renderParamLabel(
                            "forbidden（可选）",
                            "禁止通过的地址列表",
                          )}
                          <Form.Control
                            as="textarea"
                            rows={2}
                            placeholder="每行一个地址，或逗号分隔"
                            value={formData.forbidden}
                            onChange={(e) =>
                              onChange("forbidden", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={4}>
                        <Form.Group>
                          {renderParamLabel(
                            "depth",
                            "搜索深度限制（跳数），从起始节点开始的最大跳数",
                          )}
                          <Form.Control
                            type="number"
                            value={formData.depth}
                            onChange={(e) => onChange("depth", e.target.value)}
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={4}>
                        <Form.Group>
                          {renderParamLabel(
                            "activate_threshold",
                            "激活阈值（美元），低于此金额的转账将被忽略",
                          )}
                          <Form.Control
                            type="number"
                            step="0.01"
                            value={formData.activate_threshold}
                            onChange={(e) =>
                              onChange("activate_threshold", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={4}>
                        <Form.Group>
                          {renderParamLabel(
                            "age_limit",
                            "年龄限制（跳数），资金从源地址传播的最大跳数",
                          )}
                          <Form.Control
                            type="number"
                            value={formData.age_limit}
                            onChange={(e) =>
                              onChange("age_limit", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                      <Col md={12}>
                        <Form.Group>
                          {renderParamLabel(
                            "label_limit",
                            "标签数量限制，每个节点最多保留的标签数量",
                          )}
                          <Form.Control
                            type="number"
                            value={formData.label_limit}
                            onChange={(e) =>
                              onChange("label_limit", e.target.value)
                            }
                            disabled={submitting || polling}
                          />
                        </Form.Group>
                      </Col>
                    </Row>

                    {error && (
                      <Alert variant="danger" className="mt-3 mb-0">
                        {error}
                      </Alert>
                    )}

                    <div className="d-grid mt-3">
                      <Button
                        variant="primary"
                        type="submit"
                        disabled={submitting || polling}
                      >
                        {submitting ? (
                          <>
                            <Spinner
                              as="span"
                              animation="border"
                              size="sm"
                              className="me-2"
                            />
                            提交中...
                          </>
                        ) : (
                          "提交追踪任务"
                        )}
                      </Button>
                    </div>
                  </Form>
                </Card.Body>
              </Card>
            </Col>

            <Col xs={12}>
              {!task && (
                <Card>
                  <Card.Body className="text-center py-5">
                    <div style={{ fontSize: "72px", opacity: 0.2 }}>🧭</div>
                    <h5 className="text-muted mt-3">填写参数并提交追踪任务</h5>
                    <p className="text-secondary mb-0">
                      任务将异步执行，系统自动轮询状态并展示结果
                    </p>
                  </Card.Body>
                </Card>
              )}

              {task && (
                <Card>
                  <Card.Header className="d-flex justify-content-between align-items-center">
                    <div>
                      <h5 className="mb-0">任务状态</h5>
                      <small className="text-muted">{task.request_id}</small>
                    </div>
                    <Badge bg={getTaskVariant(task.status)}>
                      {task.status}
                    </Badge>
                  </Card.Header>
                  <Card.Body>
                    <Alert
                      variant={getTaskVariant(task.status)}
                      className="mb-3"
                    >
                      <div>
                        <strong>created_at:</strong> {task.created_at || "-"}
                      </div>
                      <div>
                        <strong>started_at:</strong> {task.started_at || "-"}
                      </div>
                      <div>
                        <strong>completed_at:</strong>{" "}
                        {task.completed_at || "-"}
                      </div>
                      {polling && (
                        <div className="mt-2">
                          <Spinner
                            as="span"
                            animation="border"
                            size="sm"
                            className="me-2"
                          />
                          正在轮询任务状态（每 3 秒）
                        </div>
                      )}
                    </Alert>

                    {(task.status === "failed" || task.status === "error") && (
                      <Alert variant="danger">
                        {task.error || "任务执行失败"}
                      </Alert>
                    )}

                    {task.status === "completed" && renderResult(task.result)}
                  </Card.Body>
                </Card>
              )}
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
};

export default MoneyLaunderingTrace;
