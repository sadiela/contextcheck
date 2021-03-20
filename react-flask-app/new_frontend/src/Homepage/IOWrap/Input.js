import React, { Component } from 'react';
import Alert from 'react-bootstrap/Alert';

export default class Input extends Component {
    constructor(props){
        super(props);
        this.state = {
            input_type: this.props.input_type,
            input: '',
            error: '',
        }
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleBack = this.handleBack.bind(this);
        this.handleError = this.handleError.bind(this);
    }
    handleError(){
        if(this.state.error === ''){
            return(<div></div>)
        }else{
            return(
                <p className='error'>{this.state.error}</p>
            )
        }
    }
    handleChange(event) {
        this.setState({input: event.target.value});
    }
    handleBack(){
        this.props.ValToType();
    }
    handleSubmit() {
        const input = this.state.input;
        if(this.state.input_type === 'url'){ 
            // Submit a URL
            // ***TODO: Check if it is a valid URL***
            // ***TODO: Check if it is one of our supported websites***
            this.props.handleURLSubmit(input);
        } else { 
            // Submit a Plaintext
            if(input === ''){
                this.setState({error: 'Input cannot be blank.'});
                return;
            } else {
                this.props.handlePTSubmit(input);
                this.setState({error: ''});
            }
        }
    }
    // Need a dropdown for URL input containing all supported articles
    render() {
        return (
            <div className='input-step'>
                <Alert variant='info'>Enter your <strong>{this.state.input_type}</strong> below.</Alert>
                <form>
                    <div>
                        <input style={{ width: '80%' }} type="text" id="text" name="input-type" onChange={this.handleChange} value={this.state.input} placeholder="Type / Paste here"/>
                    </div>
                    <span className='input-button-row'>
                        <button className='back-button' onClick={this.handleBack}>Back</button>
                        <button className='next-button' onClick={this.handleSubmit}>Submit</button>
                    </span>
                </form>
                {this.handleError()}
            </div>
        )
    }
}