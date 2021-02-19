import React, { useState, useEffect, useReducer } from 'react';
import './App.css';
import Form from './Form.js';
import ScraperWrapper from './Webscraper/ScraperWrapper';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>ContextCheck</h1>
      </header>
      <Form/>
      <ScraperWrapper />
    </div>
  );
}

export default App;
