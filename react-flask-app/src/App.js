import React, { useState, useEffect, useReducer } from 'react';
import './App.css';
import Form from './Form.js';
import ScraperWrapper from './Webscraper/ScraperWrapper';
//
//const formReducer = (state, event) => {
//    return {
//        ...state,
//        [event.name]: event.value
//    }
//}

function App() {
//  const [formData, setFormData] = useReducer(formReducer, {});
//  const [submitting, setSubmitting] = useState(false);
//  const [currentResponse, setCurrentResponse] = useState(0);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Team 22 - Media Bias</h1>
      </header>
      <Form/>
      <ScraperWrapper />
    </div>
  );
}

export default App;
