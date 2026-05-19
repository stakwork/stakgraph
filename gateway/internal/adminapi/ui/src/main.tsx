// SPA entry. Mounts <App/> into #root. Everything else lives in app.tsx.

import { render } from "preact";

import "./styles/base.css";
import "./styles/components.css";
import "uplot/dist/uPlot.min.css";

import { App } from "./app";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");
render(<App />, root);
