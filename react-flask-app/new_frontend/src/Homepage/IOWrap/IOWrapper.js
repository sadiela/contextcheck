import React, { Component } from 'react';
import StepZero from './IOSwitch';
import StepOne from './Input';
import Spinner from 'react-bootstrap/Spinner';

export default class IOWrapper extends Component {
    constructor(props){
        super(props);
        this.state={
            step: 0,
            input_type: '',
        }
        this.TypeToVal = this.TypeToVal.bind(this);
        this.ValToType = this.ValToType.bind(this);
        this.handleURLSubmit = this.handleURLSubmit.bind(this);
        this.handlePTSubmit = this.handlePTSubmit.bind(this);
    }
    handlePTSubmit = (text) => {
        // Passes to Homepage to make the axios call
        this.props.handlePTSubmit(text);
        this.setState({step: 0});
    }
    handleURLSubmit = (url) => {
        // Passes to Homepage to make the axios call
        this.props.handleURLSubmit(url);
        this.setState({step: 0});
    }
    TypeToVal = (type) => {
        // When the user hits the next button on the input type step
        console.log(type);
        this.setState({input_type: type});
        this.setState({step: 1});
    }
    ValToType() {
        // When the user hits the back button on the input text step
        this.setState({step: 0});
    }
    // <p className='loader'>&#9862;</p>
    render() {
        if(this.props.loading){
            return(
                <div className='input-step'>
                    <p style={{fontSize: '40px'}}><strong>Loading...</strong></p>
                    <Spinner animation="border" variant="info" />
                </div>
            )
        } else if (this.props.error != ''){
            return(
                <div className='input-step'>
                    <p style={{fontSize: '40px', color:'red'}}><strong>Ruh roh... something went wrong.</strong></p>
                    <p style={{fontSize: '20px', color:'red'}}>{this.props.error}</p>
                    <p style={{fontSize: '15px'}}>Refresh the page and try again. Sorry for the trouble! If the issue persists, take a screenshot and send it to beton@bu.edu (if you have a spare moment).</p>
                </div>
            )
        } else {
            if(this.state.step === 0){
                return(
                    <StepZero 
                        TypeToVal={this.TypeToVal}
                    />
                )
            } else if(this.state.step === 1){
                return(
                    <StepOne 
                        input_type={this.state.input_type}
                        ValToType={this.ValToType}
                        handleURLSubmit={this.handleURLSubmit}
                        handlePTSubmit={this.handlePTSubmit}
                    />
                )
            }
            // Could add a step 2 here for the interactive part (ask user how biased they think it is)
        }
    }
}