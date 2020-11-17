import React, { useState, useEffect, useReducer } from 'react';
import logo from './logo.svg';
import './App.css';
import Form from './Form.js';
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
    </div>
  );
}

export default App;
