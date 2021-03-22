import React, { Component } from 'react';
import Alert from 'react-bootstrap/Alert';


export default class IOSwitch extends Component {
    constructor(props){
        super(props);
        this.state = {
            input_type: '',
            error: ''
        }
        this.handleChange = this.handleChange.bind(this);
        this.submit = this.submit.bind(this);
        this.getError = this.getError.bind(this);
    }
    handleChange(event) {
        this.setState({input_type: event.target.value});
    }
    getError() {
        if(this.state.error === ''){
            return (<div></div>)
        } else {
            return (<p className='error'>
                {this.state.error}
            </p>)
        }
    }
    submit() {
        if(this.state.input_type === ''){
            this.setState({error: 'You must select a type.'})
        } else {
            this.setState({error: ''});
            this.props.TypeToVal(this.state.input_type);
        }
        
    }
    render() {
        return (
            <div className='input-step'>
                <Alert variant='info'>Would you like to submit an article link or plain text?</Alert>
                <form onSubmit={this.submit}>
                    <span className='radio-group' onChange={this.handleChange}>
                        <span>
                            <input type="radio" id="url" name="input-type" value="url"/>
                            <label className='radio-label' htmlFor="url">URL</label><br />
                        </span>
                        <span>
                            <input type="radio" id="plaintext" name="input-type" value="plaintext"/>
                            <label className='radio-label' htmlFor="plaintext">Plain Text</label><br />
                        </span>
                    </span>
                    <input className='next-button' type='submit' value="Next"></input>
                </form>
                {this.getError()}
            </div>
        )
    }
}