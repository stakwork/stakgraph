import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import "./App.css";
import People from "./components/People";
import NewPerson from "./components/NewPerson";

export const AppName: string = "My React App";
export const hostPort: string = "http://localhost:5002";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>My React App</h1>
      </header>
      <Router>
        <Routes>
          <Route path="/people" element={<People />} />
          <Route path="/new-person" element={<NewPerson />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
