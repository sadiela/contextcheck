import React, { Component } from 'react';
import Alert from 'react-bootstrap/Alert';
import URLDropdown from './SupportedDropdown';
import isValidURL from './PossibleURLs';

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
            if(isValidURL(input)){
                this.props.handleURLSubmit(input);
                this.setState({error: ''});
            } else {
                this.setState({error: 'Invalid URL Error'})
            }
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
    getDropdown() {
        if(this.state.input_type === 'url'){
            return(
                <URLDropdown/>            
            )
        } else return(<></>)
    }
    render() {
        return (
            <div className='input-step'>
                <Alert variant='info'>Enter your <strong>{this.state.input_type}</strong> below.</Alert>
                <form>
                    <div>
                        <input style={{ width: '80%' }} type="text" id="text" name="input-type" onChange={this.handleChange} value={this.state.input} placeholder="Type / Paste here"/>
                    </div>
                    {this.getDropdown()}
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