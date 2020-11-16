import React, { useState, useEffect, useReducer } from 'react';
import logo from './logo.svg';
import './App.css';

const formReducer = (state, event) => {
    return {
        ...state,
        [event.target.name]: event.target.value
    }
}

function App() {
  const [formData, setFormData] = useReducer(formReducer, {});
  const [submitting, setSubmitting] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  useEffect(() => {
		  fetch('/time').then(res => res.json()).then(data => {
				  setCurrentTime(data.time);
		  });
  }, []);
  const handleSubmit = event => {
    event.preventDefault();
    setSubmitting(true);
    setTimeout(() => {
        setSubmitting(false);
    }, 3000)
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Input Text Below</h1>
		<p>The current time is {currentTime}.</p>
      </header>
      {submitting &&
        <div>Submitting...</div>
      }
      <form className="form" onSubmit={handleSubmit}>
        <fieldset>
            <label>
                <p>Text Input</p>
                <input defaultValue="Enter Text Here" onChange={setFormData}/>
            </label>
        </fieldset>
        <button type="submit">Submit</button>
      </form>
    </div>
  );
}

export default App;
