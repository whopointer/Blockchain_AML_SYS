import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Alert, Button, Spinner, Badge } from 'react-bootstrap';
import { api, HealthResponse, ModelInfo, StatisticsResponse, getModelDisplayName, getModelColor } from '../services/api';

interface SupportedModel {
  id: string;
  name: string;
  description: string;
}

interface DashboardProps {
  onModelSwitch?: (modelType: string) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onModelSwitch }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [statistics, setStatistics] = useState<StatisticsResponse | null>(null);
  const [supportedModels, setSupportedModels] = useState<SupportedModel[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [switchingModel, setSwitchingModel] = useState<string | null>(null);
  const [error, setError] = useState<string>('');
  const [switchMessage, setSwitchMessage] = useState<string>('');

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    setError('');

    try {
      const [healthData, modelData, statsData, modelsData] = await Promise.all([
        api.healthCheck(),
        api.getModelInfo().catch(() => null),
        api.getStatistics(),
        api.getModels().catch(() => ({ supported_models: [], descriptions: {} })),
        fetch('http://localhost:8000/api/v1/mode').then(res => res.json()).catch(() => ({ mode: 'single' }))  // 获取当前模式
      ]);

      setHealth(healthData);
      setModelInfo(modelData);
      setStatistics(statsData);

      // 解析支持的模型列表
      const models: SupportedModel[] = [];
      if ('supported_models' in modelsData) {
        const descriptions = (modelsData as any).descriptions || {};
        for (const id of (modelsData as any).supported_models || []) {
          models.push({
            id,
            name: getModelDisplayName(id),
            description: descriptions[id] || ''
          });
        }
      }
      setSupportedModels(models);

    } catch (err: any) {
      setError(err.response?.data?.error || '加载仪表板数据失败');
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchModel = async (modelType: string) => {
    if (modelType === health?.model_type || switchingModel) return;

    setSwitchingModel(modelType);
    setSwitchMessage('');
    setError('');

    try {
      const result = await api.switchModel(modelType);
      setSwitchMessage(result.message);
      // 重新加载数据
      await loadDashboardData();
      // 通知父组件模型已切换
      onModelSwitch?.(modelType);
    } catch (err: any) {
      setError(err.response?.data?.detail || '模型切换失败');
    } finally {
      setSwitchingModel(null);
    }
  };

  if (loading) {
    return (
      <div className="text-center p-4">
        <Spinner animation="border" />
        <p className="mt-2">加载中...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="text-center mb-4">
        <h2>系统仪表板</h2>
        <p className="text-secondary">区块链AML反洗钱系统实时监控</p>
      </div>

      {switchMessage && (
        <Alert variant="success" className="mb-4">
          ✅ {switchMessage}
        </Alert>
      )}

      {error && <Alert variant="danger">{error}</Alert>}

      {/* 当前模型信息 */}
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">🤖 检测模型</h5>
        </Card.Header>
        <Card.Body>
          <div className="p-3 rounded border border-primary bg-light">
            <div className="d-flex justify-content-between align-items-center">
              <div>
                <h6 className="mb-1">
                  <span className="me-2">🛡️</span>
                  DGI + GIN + Random Forest
                </h6>
                <small className="text-muted">基于图神经网络的反洗钱检测模型</small>
              </div>
              <Badge bg="primary">当前使用</Badge>
            </div>
          </div>
        </Card.Body>
      </Card>

      {/* 支持的模型列表 */}
      {supportedModels.length > 0 && (
        <Card className="mb-4">
          <Card.Header>
            <h5 className="mb-0">🔧 选择检测模型</h5>
          </Card.Header>
          <Card.Body>
            <Row className="g-3">
              {supportedModels.map((model) => (
                <Col md={6} key={model.id}>
                  <div 
                    className={`p-3 rounded border cursor-pointer transition-all ${
                      model.id === health?.model_type 
                        ? 'border-primary bg-light' 
                        : 'border-secondary hover-border-primary'
                    }`}
                    style={{ 
                      cursor: model.id === health?.model_type ? 'default' : 'pointer',
                      opacity: switchingModel === model.id ? 0.5 : 1,
                      transition: 'all 0.2s ease'
                    }}
                    onClick={() => model.id !== health?.model_type && handleSwitchModel(model.id)}
                  >
                    <div className="d-flex justify-content-between align-items-center">
                      <div>
                        <h6 className="mb-1">
                          <span 
                            className="me-2"
                            style={{ 
                              display: 'inline-block',
                              width: '12px',
                              height: '12px',
                              borderRadius: '50%',
                              background: getModelColor(model.id)
                            }}
                          />
                          {model.name}
                        </h6>
                        <small className="text-muted">{model.description}</small>
                      </div>
                      {model.id === health?.model_type ? (
                        <Badge bg="primary">当前使用</Badge>
                      ) : switchingModel === model.id ? (
                        <Spinner animation="border" size="sm" />
                      ) : (
                        <Button variant="outline-primary" size="sm">
                          切换
                        </Button>
                      )}
                    </div>
                  </div>
                </Col>
              ))}
            </Row>
          </Card.Body>
        </Card>
      )}

      <Row className="g-4 mb-4">
        <Col xl={4} lg={6} md={12}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <div style={{ 
                    width: '48px', 
                    height: '48px', 
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontSize: '24px'
                  }}>
                    🖥️
                  </div>
                </div>
                <div>
                  <h5 className="mb-0">系统状态</h5>
                  <small className="text-muted">System Status</small>
                </div>
              </div>
            </Card.Header>
            <Card.Body>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">运行状态</span>
                <span className={`badge ${
                  health?.status === 'healthy' ? 'bg-success' : 'bg-danger'
                }`}>
                  {health?.status === 'healthy' ? '✓ 正常运行' : '✗ 系统异常'}
                </span>
              </div>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">系统版本</span>
                <span className="text-primary">{statistics?.version}</span>
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span className="text-secondary">当前模型</span>
                <span className="badge bg-info">
                  {getModelDisplayName(health?.model_type || 'unknown')}
                </span>
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col xl={4} lg={6} md={12}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <div style={{ 
                    width: '48px', 
                    height: '48px', 
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #13B497 0%, #59D4A4 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontSize: '24px'
                  }}>
                    🤖
                  </div>
                </div>
                <div>
                  <h5 className="mb-0">模型详情</h5>
                  <small className="text-muted">Model Details</small>
                </div>
              </div>
            </Card.Header>
            <Card.Body>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">模型加载</span>
                <span className={`badge ${
                  health?.model_loaded ? 'bg-success' : 'bg-warning'
                }`}>
                  {health?.model_loaded ? '✓ 已加载' : '⚠ 未加载'}
                </span>
              </div>
              {modelInfo && (
                <>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">模型版本</span>
                    <span className="text-primary">{modelInfo.model_version || '-'}</span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center">
                    <span className="text-secondary">加载时间</span>
                    <small className="text-muted">
                      {modelInfo.loaded_at ? new Date(modelInfo.loaded_at).toLocaleString() : '-'}
                    </small>
                  </div>
                </>
              )}
            </Card.Body>
          </Card>
        </Col>

        <Col xl={4} lg={6} md={12}>
          <Card className="h-100">
            <Card.Header>
              <div className="d-flex align-items-center">
                <div className="me-3">
                  <div style={{ 
                    width: '48px', 
                    height: '48px', 
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #FFA726 0%, #FF7043 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontSize: '24px'
                  }}>
                    📈
                  </div>
                </div>
                <div>
                  <h5 className="mb-0">性能指标</h5>
                  <small className="text-muted">Performance Metrics</small>
                </div>
              </div>
            </Card.Header>
            <Card.Body>
              {modelInfo?.performance_metrics ? (
                <>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">准确率</span>
                    <span className="text-success font-weight-bold">
                      {((modelInfo.performance_metrics.accuracy ?? 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">AUC</span>
                    <span className="text-info font-weight-bold">
                      {((modelInfo.performance_metrics.auc ?? 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mb-3">
                    <span className="text-secondary">AP</span>
                    <span className="text-warning font-weight-bold">
                      {((modelInfo.performance_metrics.average_precision ?? 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center">
                    <span className="text-secondary">F1分数</span>
                    <span className="text-primary font-weight-bold">
                      {((modelInfo.performance_metrics.f1_score ?? 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                </>
              ) : (
                <div className="text-center py-3">
                  <div style={{ fontSize: '48px', opacity: 0.3 }}>📊</div>
                  <p className="text-muted mt-2">暂无性能数据</p>
                </div>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Card>
        <Card.Header>
          <div className="d-flex align-items-center">
            <div className="me-3">
              <div style={{ 
                width: '48px', 
                height: '48px', 
                borderRadius: '12px',
                background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '24px'
              }}>
                ℹ️
              </div>
            </div>
            <div>
              <h5 className="mb-0">系统信息</h5>
              <small className="text-muted">System Information</small>
            </div>
          </div>
        </Card.Header>
        <Card.Body>
          <Row className="g-4">
            <Col md={6}>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">API端点</span>
                <code>http://localhost:8001</code>
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span className="text-secondary">数据加载</span>
                <span className={`badge ${
                  health?.data_loaded ? 'bg-success' : 'bg-warning'
                }`}>
                  {health?.data_loaded ? '✓ 已加载' : '⚠ 未加载'}
                </span>
              </div>
            </Col>
            <Col md={6}>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span className="text-secondary">缓存状态</span>
                <span className={`badge ${
                  health?.cache_built ? 'bg-success' : 'bg-warning'
                }`}>
                  {health?.cache_built ? '✓ 已构建' : '⚠ 未构建'}
                </span>
              </div>
              <Button variant="outline-primary" onClick={loadDashboardData} className="w-100">
                🔄 刷新数据
              </Button>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
};

export default Dashboard;
