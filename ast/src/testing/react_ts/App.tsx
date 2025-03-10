import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './App.css';
import People from './components/People';
import NewPerson from './components/NewPerson';

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
