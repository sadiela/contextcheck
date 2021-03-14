import React, { useState, useEffect, useReducer } from 'react';
import './App.css';
import PTWrapper from './Plaintextscraper/PTWrapper';
import ScraperWrapper from './Webscraper/ScraperWrapper';
import logo from './logoContext.png'
function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} alt="ContextCheck"></img>
      </header>
      <PTWrapper />
      <ScraperWrapper />
    </div>
  );
}

export default App;
