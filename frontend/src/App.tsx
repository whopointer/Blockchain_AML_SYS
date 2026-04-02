import React, { useState } from "react";
import { Container, Nav, Navbar, Tab, Tabs, Row, Col } from "react-bootstrap";
import {
  Routes,
  Route,
  Navigate,
  useLocation,
  Link,
} from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

import Dashboard from "./components/Dashboard";
import PredictionForm from "./components/PredictionForm";
import ResultsTable from "./components/ResultsTable";
import BatchAnalysis from "./components/BatchAnalysis";
import MoneyLaunderingTrace from "./components/MoneyLaunderingTrace";
import TransactionGraph from "./components/TransactionGraph";
import CaseDetails from "./components/CaseDetails";
import PathTracking from "./components/PathTracking";
import { PredictionResponse } from "./services/api";

// 自定义导航链接组件，用于激活状态样式
const CustomNavLink = ({
  children,
  to,
  ...props
}: {
  children: React.ReactNode;
  to: string;
  [key: string]: any;
}) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Nav.Link as={Link} to={to} className={isActive ? "active" : ""} {...props}>
      {children}
    </Nav.Link>
  );
};

// Dashboard页面组件，包含Tabs
const DashboardPage = () => {
  const [predictionResults, setPredictionResults] =
    useState<PredictionResponse | null>(null);

  const handlePredictionComplete = (results: PredictionResponse) => {
    setPredictionResults(results);
  };

  return (
    <Tab.Container defaultActiveKey="dashboard">
      <Tabs id="main-tabs" className="mb-4" fill justify>
        <Tab eventKey="dashboard" title="🎯 系统仪表板">
          <Dashboard />
        </Tab>
        <Tab eventKey="prediction" title="🔍 交易异常检测">
          <Row>
            <Col lg={5} className="mb-4">
              <PredictionForm onPredictionComplete={handlePredictionComplete} />
            </Col>
            <Col lg={7}>
              <ResultsTable results={predictionResults} />
            </Col>
          </Row>
        </Tab>
        <Tab eventKey="batch" title="📊 批量分析">
          <BatchAnalysis />
        </Tab>
        <Tab eventKey="trace" title="🔗 洗钱路径追踪">
          <MoneyLaunderingTrace />
        </Tab>
      </Tabs>
    </Tab.Container>
  );
};

function App() {
  return (
    <div className="App">
      <Navbar expand="lg" className="fixed-top">
        <Container>
          <Navbar.Brand as={Link} to="/dashboard">
            区块链AML反洗钱系统
          </Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <CustomNavLink to="/dashboard">系统仪表板</CustomNavLink>
              <CustomNavLink to="/prediction">交易检测</CustomNavLink>
              <CustomNavLink to="/batch">批量分析</CustomNavLink>
              <CustomNavLink to="/transaction-graph">交易图谱</CustomNavLink>
              <CustomNavLink to="/path-tracking">路径追踪</CustomNavLink>
              <CustomNavLink to="/case-details">案件详情</CustomNavLink>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container fluid className="mt-5 pt-4">
        <Routes>
          {/* 仪表板相关页面保持原有宽度限制 */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route
            path="/dashboard"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />
          <Route
            path="/prediction"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />
          <Route
            path="/batch"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />
          <Route
            path="/trace"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={11} md={12}>
                  <DashboardPage />
                </Col>
              </Row>
            }
          />

          <Route
            path="/transaction-graph"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={12} md={12}>
                  <TransactionGraph />
                </Col>
              </Row>
            }
          />
          <Route
            path="/transaction-graph/:crypto/:address"
            element={
              <Row className="justify-content-center">
                <Col xl={12} lg={12} md={12}>
                  <TransactionGraph />
                </Col>
              </Row>
            }
          />
          <Route
            path="/path-tracking"
            element={
              <Row className="justify-content-center">
                <Col xl={10} lg={12} md={12}>
                  <PathTracking />
                </Col>
              </Row>
            }
          />
          <Route
            path="/path-tracking/:crypto"
            element={
              <Row className="justify-content-center">
                <Col xl={12} lg={12} md={12}>
                  <PathTracking />
                </Col>
              </Row>
            }
          />

          {/* 案件详情页面使用全屏宽度 */}
          <Route
            path="/case-details"
            element={
              <Row className="justify-content-center">
                <Col xl={20} lg={18} md={14}>
                  <CaseDetails />
                </Col>
              </Row>
            }
          />
          <Route
            path="/case-details/*"
            element={
              <Row className="justify-content-center">
                <CaseDetails />
              </Row>
            }
          />

          {/* 处理无效路径 */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Container>
    </div>
  );
}

export default App;
