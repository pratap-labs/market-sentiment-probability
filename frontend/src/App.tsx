import { BrowserRouter, NavLink, Route, Routes, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import PortfolioDashboard from "./pages/PortfolioDashboard";
import Equities from "./pages/Equities";
import RiskBuckets from "./pages/RiskBuckets";
import RiskBucketsPortfolio from "./pages/RiskBucketsPortfolio";
import RiskBucketsBucket from "./pages/RiskBucketsBucket";
import RiskBucketsTrade from "./pages/RiskBucketsTrade";
import RiskBucketsMeta from "./pages/RiskBucketsMeta";
import RiskBucketsSettings from "./pages/RiskBucketsSettings";
import PreTradeStress from "./pages/PreTradeStress";
import TradeSelector from "./pages/TradeSelector";
import MarketRegime from "./pages/MarketRegime";
import HistoricalPerformance from "./pages/HistoricalPerformance";
import ProductOverview from "./pages/ProductOverview";
import DataSource from "./pages/DataSource";
import RiskBucketsSpotAnalysis from "./pages/RiskBucketsSpotAnalysis";
import { PortfolioProvider } from "./state/PortfolioContext";
import { NotificationProvider } from "./state/NotificationContext";
import AuthGate from "./components/AuthGate";
import { ControlsProvider } from "./state/ControlsContext";
import ControlsBar from "./components/ControlsBar";
import LiveTicker from "./components/LiveTicker";

const navItems = [
  { label: "Login", to: "/login" },
  { label: "Portfolio", to: "/portfolio" },
  { label: "Equities", to: "/equities" },
  { label: "Spot Analysis", to: "/spot-analysis" },
  { label: "Risk Buckets", to: "/risk-buckets" },
  { label: "Pre-Trade", to: "/pre-trade" },
  { label: "Trade Selector", to: "/trade-selector" },
  { label: "Market Regime", to: "/market-regime" },
  { label: "Historical", to: "/historical" },
  { label: "Data Source", to: "/data-source" },
  { label: "Settings", to: "/settings" }
];

export default function App() {
  return (
    <BrowserRouter>
      <NotificationProvider>
        <ControlsProvider>
          <PortfolioProvider>
            <AuthGate>
              <div className="app-shell">
                <aside className="sidebar">
                  <div className="brand">
                    <img src="/gammashield-logo.png" alt="GammaShield" className="brand-logo" />
                    <div className="brand-subtitle">Options Risk Engine</div>
                  </div>
                  <nav className="nav">
                    {navItems.map((item) => (
                      <NavLink key={item.to} to={item.to} className={({ isActive }) => (isActive ? "active" : "")}
                      >
                        {item.label}
                      </NavLink>
                    ))}
                  </nav>
                </aside>
                <div className="main">
                  <header className="topbar">
                    <LiveTicker />
                  </header>
                  <ControlsBar />
                  <main className="content">
                    <Routes>
                      <Route path="/" element={<Login />} />
                      <Route path="/login" element={<Login />} />
                      <Route path="/portfolio" element={<PortfolioDashboard />} />
                      <Route path="/equities" element={<Equities />} />
                      <Route path="/spot-analysis" element={<RiskBucketsSpotAnalysis />} />
                      <Route path="/risk-buckets" element={<RiskBuckets />}>
                        <Route index element={<Navigate to="portfolio" replace />} />
                        <Route path="portfolio" element={<RiskBucketsPortfolio />} />
                        <Route path="bucket" element={<RiskBucketsBucket />} />
                        <Route path="trade" element={<RiskBucketsTrade />} />
                        <Route path="spot-analysis" element={<RiskBucketsSpotAnalysis />} />
                        <Route path="meta" element={<RiskBucketsMeta />} />
                      </Route>
                      <Route path="/pre-trade" element={<PreTradeStress />} />
                      <Route path="/trade-selector" element={<TradeSelector />} />
                      <Route path="/market-regime" element={<MarketRegime />} />
                      <Route path="/historical" element={<HistoricalPerformance />} />
                      <Route path="/product" element={<ProductOverview />} />
                      <Route path="/derivatives" element={<RiskBucketsSpotAnalysis />} />
                      <Route path="/data-source" element={<DataSource />} />
                      <Route path="/settings" element={<RiskBucketsSettings />} />
                    </Routes>
                  </main>
                </div>
              </div>
            </AuthGate>
          </PortfolioProvider>
        </ControlsProvider>
      </NotificationProvider>
    </BrowserRouter>
  );
}
