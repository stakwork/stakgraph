import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import "./App.css";
import People from "./components/People";
import NewPerson from "./components/NewPerson";
import { ExportedArrowComponent, ExportedFunctionComponent, ArrowComponent, DirectAssignmentComponent } from "./components/ComponentStyles";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>My React App</h1>
        <ExportedArrowComponent items={["Test", "Component", "Styles"]} />
        <ExportedFunctionComponent name="Test User" />
        <ArrowComponent count={5} />
        <DirectAssignmentComponent value="Direct Assignment Test" />
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
