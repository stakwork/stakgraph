import React, {useEffect} from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import "./App.css";
import People from "./components/People";
import NewPerson from "./components/NewPerson";

enum APP_NAMES{
  MyReactApp = "My React App"
  MyApp = "My App"
  MyDemoApp = "My Demo App"
}

// The name of the application
// @ast node: Var "AppName"
export const AppName: string = APP_NAMES.MyReactApp;
// @ast node: Var "hostPort"
export const hostPort: string = "http://localhost:5002";

// @ast node: Class "TestThing"
class TestThing() {
  constructor() {
    super();
  }
}

// @ast node: Function "App"
// @ast node: Page "/people"
// @ast edge: Renders -> Function "People" "People.tsx"
// @ast node: Page "/new-person"
// @ast edge: Renders -> Function "NewPerson" "NewPerson.tsx"
function App() {

  const testThing = new TestThing();

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
