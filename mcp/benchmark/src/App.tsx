import { Toaster } from "sonner";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { Analytics } from "./components/Analytics";
import { Sessions } from "./components/Sessions";

export default function App() {
  const navStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
    fontSize: "12px",
    fontWeight: 600,
    color: isActive ? "#ededed" : "#71717a",
    textDecoration: "none",
    padding: "6px 10px",
    borderRadius: "8px",
    border: `1px solid ${isActive ? "#3f3f46" : "transparent"}`,
    backgroundColor: isActive ? "#18181b" : "transparent",
  });

  return (
    <BrowserRouter basename="/sessions">
      <div
        className="h-screen flex flex-col overflow-hidden"
        style={{
          backgroundColor: "var(--background)",
          color: "var(--foreground)",
        }}
      >
        <header
          className="border-b flex items-center justify-between px-6 h-12 shrink-0"
          style={{ borderColor: "var(--border)" }}
        >
          <span className="text-sm font-semibold tracking-wide">
            stakgraph sessions
          </span>
          <nav style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <NavLink to="/" end style={navStyle}>
              Sessions
            </NavLink>
            <NavLink to="/analytics" style={navStyle}>
              Analytics
            </NavLink>
          </nav>
        </header>
        <main className="flex-1 min-h-0 overflow-hidden p-6 flex flex-col">
          <Routes>
            <Route path="/" element={<Sessions />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
        <Toaster theme="dark" />
      </div>
    </BrowserRouter>
  );
}
