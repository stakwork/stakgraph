import { Toaster } from "sonner";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Sessions } from "./components/Sessions";

export default function App() {
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
          className="border-b flex items-center px-6 h-12 shrink-0"
          style={{ borderColor: "var(--border)" }}
        >
          <span className="text-sm font-semibold tracking-wide">
            stakgraph sessions
          </span>
        </header>
        <main className="flex-1 min-h-0 overflow-hidden p-6 flex flex-col">
          <Routes>
            <Route path="/" element={<Sessions />} />
          </Routes>
        </main>
        <Toaster theme="dark" />
      </div>
    </BrowserRouter>
  );
}
