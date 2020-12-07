import axios from 'axios';
import React from 'react';
import './Form.css';
class Form extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            text: "Enter text here",
            response: "Default"
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }
    handleChange(event){
        this.setState({text: event.target.value, response: event.target.response});
    }
    handleSubmit(event){
        event.preventDefault();
        const myText = this.state.text;
        console.log("Input: " + myText);
        axios.post("/result", {myText})
            .then(res => {
                this.setState({response: res.data.text});
                console.log("Output: " + res.data.text);
            });
    }
    render(){
        return(
            <form className="form" onSubmit={this.handleSubmit}>
            <p></p>
                <label>
                    <strong>Text:</strong> <input className="text_input" key="text-input" type="text" value={this.state.text} onChange={this.handleChange} />
                </label>
                <button className="submit_button" type="submit" value="Submit">Submit</button>
                <ul>
                <li className="list_item"> Response: {this.state.response} </li>
                </ul>
            </form>
        );
    }
}
export default Form;