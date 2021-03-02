import axios from 'axios';
import React from 'react';
import './Form.css';

class Form extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            text: "Enter text here",
            response: []
        };
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleChange = this.handleChange.bind(this);
    }
    handleChange(event){
        this.setState({text: event.target.value}); //, response: event.target.response});
    }
    handleSubmit(event){
        event.preventDefault();
        this.props.handleSubmit(this.state.text);
    }
    render(){
        return(
            <div>
                <form className="form" onSubmit={this.handleSubmit}>
                <p></p>
                    <label>
                        <strong>Text:</strong> <input className="text_input" key="text-input" type="text" value={this.state.text} onChange={this.handleChange} />
                    </label>
                    <button className="submit_button" type="submit" value="Submit">Submit</button>
                </form>
                <p></p>
            </div>
        );
    }
}
export default Form;