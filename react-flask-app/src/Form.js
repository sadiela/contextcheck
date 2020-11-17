import axios from 'axios';
import React from 'react';
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
        console.log("handleChange: " + event.target.value);
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
            <form onSubmit={this.handleSubmit}>
                <label>
                    Text: <input key="text-input" type="text" value={this.state.text} onChange={this.handleChange} />
                </label>
                <input type="submit" value="Submit"/>
                <p>Response: {this.state.response}</p>
            </form>
        );
    }
}
export default Form;