import React from 'react';
import { Table, Badge, Alert } from 'react-bootstrap';
import { PredictionResponse, getModelDisplayName, getModelColor } from '../services/api';

interface ResultsTableProps {
  results: PredictionResponse | null;
}

const ResultsTable: React.FC<ResultsTableProps> = ({ results }) => {
  if (!results) {
    return null;
  }

  const getRiskBadgeVariant = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'danger';
      case 'medium': return 'warning';
      case 'low': return 'success';
      default: return 'secondary';
    }
  };

  const getSuspiciousBadgeVariant = (isSuspicious: boolean) => {
    return isSuspicious ? 'danger' : 'success';
  };

  const modelColor = getModelColor(results.model_type || '');

  return (
    <div className="results-table">
      <div className="text-center mb-4">
        <h4>📊 检测结果分析</h4>
        <p className="text-secondary">
          DGI + GIN + Random Forest 智能分析结果报告
        </p>
      </div>
      
      {/* 模型信息 */}
      <Alert 
        variant="info" 
        className="mb-4"
        style={{
          border: `1px solid ${modelColor}40`,
          background: `${modelColor}10`
        }}
      >
        <div className="d-flex align-items-center">
          <div 
            className="me-3"
            style={{
              width: '48px',
              height: '48px',
              borderRadius: '12px',
              background: modelColor,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '24px'
            }}
          >
            🤖
          </div>
          <div className="flex-grow-1">
            <h6 className="mb-1">
              检测模型: {getModelDisplayName(results.model_type || '')}
            </h6>
            <div className="d-flex gap-4 flex-wrap">
              <span>
                <strong>总交易数:</strong> {results.total_transactions}
              </span>
              <span>
                <strong>可疑交易:</strong> 
                <span className="ms-1 badge bg-danger">{results.suspicious_count}</span>
              </span>
              <span>
                <strong>可疑比例:</strong> 
                <span className="ms-1 badge bg-warning">
                  {((results.suspicious_count / results.total_transactions) * 100).toFixed(2)}%
                </span>
              </span>
            </div>
          </div>
        </div>
      </Alert>

      <div className="table-responsive">
        <Table striped hover responsive className="align-middle">
          <thead>
            <tr>
              <th style={{ width: '40%' }}>
                <span className="me-2">🔗</span>
                交易ID
              </th>
              <th style={{ width: '15%' }}>
                <span className="me-2">🚨</span>
                状态
              </th>
              <th style={{ width: '25%' }}>
                <span className="me-2">📊</span>
                置信度
              </th>
              <th style={{ width: '20%' }}>
                <span className="me-2">⚡</span>
                风险等级
              </th>
            </tr>
          </thead>
          <tbody>
            {results.results.map((result, index) => (
              <tr key={index} className={result.is_suspicious ? 'table-danger' : 'table-success'}>
                <td>
                  <code style={{ 
                    fontSize: '0.85rem',
                    wordBreak: 'break-all',
                    display: 'block',
                    padding: '0.5rem',
                    background: 'rgba(26, 32, 53, 0.5)',
                    borderRadius: '6px',
                    border: '1px solid var(--card-border)'
                  }}>
                    {result.tx_id}
                  </code>
                </td>
                <td>
                  <Badge 
                    bg={getSuspiciousBadgeVariant(result.is_suspicious)}
                    className="px-3 py-2"
                  >
                    <span className="me-1">
                      {result.is_suspicious ? '🚨' : '✅'}
                    </span>
                    {result.is_suspicious ? '可疑' : '正常'}
                  </Badge>
                </td>
                <td>
                  <div className="confidence-score">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <small className="text-muted">置信度</small>
                      <small className="font-weight-bold">
                        {result.confidence != null && !isNaN(result.confidence)
                          ? `${(result.confidence * 100).toFixed(1)}%`
                          : 'N/A'}
                      </small>
                    </div>
                    <div className="progress" style={{ height: '8px' }}>
                      <div
                        className={`progress-bar ${
                          result.confidence > 0.8 ? 'bg-success' : 
                          result.confidence > 0.6 ? 'bg-warning' : 'bg-danger'
                        }`}
                        role="progressbar"
                        style={{ 
                          width: result.confidence != null && !isNaN(result.confidence)
                            ? `${result.confidence * 100}%`
                            : '0%',
                          transition: 'width 0.6s ease'
                        }}
                        aria-valuenow={result.confidence != null && !isNaN(result.confidence) 
                          ? result.confidence * 100 
                          : 0}
                        aria-valuemin={0}
                        aria-valuemax={100}
                      />
                    </div>
                  </div>
                </td>
                <td>
                  <Badge 
                    bg={getRiskBadgeVariant(result.risk_level)}
                    className="px-3 py-2"
                  >
                    <span className="me-1">
                      {result.risk_level === 'high' ? '🔴' : 
                       result.risk_level === 'medium' ? '🟡' : '🟢'}
                    </span>
                    {result.risk_level === 'high' ? '高风险' : 
                     result.risk_level === 'medium' ? '中风险' : '低风险'}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </div>

      {results.suspicious_count > 0 && (
        <Alert variant="warning" className="mt-4">
          <div className="d-flex align-items-center">
            <span className="me-3" style={{ fontSize: '24px' }}>⚠️</span>
            <div>
              <strong>风险提醒：</strong>
              检测到 {results.suspicious_count} 笔可疑交易，建议进一步人工审核。
              可疑交易可能涉及洗钱、欺诈或其他非法活动。
            </div>
          </div>
        </Alert>
      )}
    </div>
  );
};

export default ResultsTable;