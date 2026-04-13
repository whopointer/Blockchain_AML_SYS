import React from "react";
import { Container, Nav, Navbar, Tab, Tabs, Row, Col } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

import Dashboard from "./components/Dashboard";
import PredictionForm from "./components/PredictionForm";
import ResultsTable from "./components/ResultsTable";
import BatchAnalysis from "./components/BatchAnalysis";
import MoneyLaunderingTrace from "./components/MoneyLaunderingTrace";
import AddressDetectionPanel from "./components/detection/AddressDetectionPanel";
import TransactionGraph from "./components/TransactionGraph";
import CaseDetails from "./components/CaseDetails";
import PathTracking from "./components/PathTracking";
import { PredictionResponse } from "./services/api";
import MonitoredAddressList from "./components/MonitoredAddresses"
import ReportList from "./components/Reports"
import AlertList from  "./components/Alerts"

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
      <Helmet>
        <title>系统仪表板 - 区块链AML反洗钱系统</title>
      </Helmet>
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
        <Tab eventKey="monitor" title="🔗 监控地址列表">
          <MonitoredAddressList />
        </Tab>
        <Tab eventKey="reports" title="🔗 报告列表">
          <ReportList />
        </Tab>
        <Tab eventKey="alerts" title="🔗 告警列表">
          <AlertList />
        </Tab>
      </Tabs>
    </Tab.Container>
  );
};

function App() {
  const handleModelSwitch = (modelType: string) => {
    // 用于 Dashboard 的模型切换回调
    console.log("Model switched to:", modelType);
  };

  return (
    <div className="App">
      <Navbar expand="lg" className="fixed-top">
        <Container>
          <Navbar.Brand href="#home">区块链AML反洗钱系统</Navbar.Brand>
          <Navbar.Toggle aria-controls="basic-navbar-nav" />
          <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link href="#dashboard">系统仪表板</Nav.Link>
              <Nav.Link href="#detection">地址检测</Nav.Link>
              <Nav.Link href="#trace">路径追踪</Nav.Link>
              <Nav.Link href="#graph">交易图谱</Nav.Link>
              <Nav.Link href="#cases">案件详情</Nav.Link>
            </Nav>
          </Navbar.Collapse>
        </Container>
      </Navbar>

      <Container fluid className="mt-5 pt-4">
        <Row className="justify-content-center">
          <Col xl={10} lg={11} md={12}>
            <Tabs
              defaultActiveKey="dashboard"
              id="main-tabs"
              className="mb-4"
              fill
              justify
            >
              <Tab eventKey="dashboard" title="🎯 系统仪表板">
                <Dashboard
                  onModelSwitch={handleModelSwitch}
                />
              </Tab>
              <Tab eventKey="detection" title="🔍 地址检测">
                <AddressDetectionPanel />
              </Tab>
              <Tab eventKey="trace" title="🔗 洗钱路径追踪">
                <MoneyLaunderingTrace />
              </Tab>
              <Tab eventKey="graph" title="📈 交易图谱">
                <TransactionGraph />
              </Tab>
              <Tab eventKey="cases" title="📋 案件详情">
                <CaseDetails />
              </Tab>
            </Tabs>
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;