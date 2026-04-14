import React from "react";
import { Container, Nav, Navbar, Tab, Tabs, Row, Col } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

import Dashboard from "./components/Dashboard";
import ResultsTable from "./components/ResultsTable";
import MoneyLaunderingTrace from "./components/MoneyLaunderingTrace";
import AddressDetectionPanel from "./components/detection/AddressDetectionPanel";
import TransactionGraph from "./components/TransactionGraph";
import CaseDetails from "./components/CaseDetails";
import { PredictionResponse } from "./services/api";
import MonitoredAddressList from "./components/MonitoredAddresses"
import ReportList from "./components/Reports"
import AlertList from  "./components/Alerts"


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
              <Tab eventKey="monitor" title="👁️‍🗨️ 监控地址">
                <MonitoredAddressList />
              </Tab>
              <Tab eventKey="reports" title="📄 报告查看">
                <ReportList />
              </Tab>
              <Tab eventKey="alerts" title="⚠️ 告警查询">
                <AlertList />
              </Tab>
            </Tabs>
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;