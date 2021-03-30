import React, { Component } from 'react';
import Header from './Header/HeaderWrapper';
import InputWrapper from './InputWrapper/InputWrapper';
import Results from './Results/ResultsWrapper';
import axios from 'axios';

export default class Homepage extends Component {
    constructor(props){
        super(props);
        this.state = {
            loading: false, // To render the loading indicator
            output: { // The output object that is returned from backend server
                author: [],
                bias_results: {},
                date: "",
                feedText: "",
                related: {},
                title: ""
            },
            results: false, // Whether or not results have been fetched
            input_type: '', 
            error: '',
        }
        this.handlePTSubmit = this.handlePTSubmit.bind(this);
        this.handleURLSubmit = this.handleURLSubmit.bind(this);
    }
    /**
     * This method handles the plaintext submit
     * It makes a POST request using the inputted text as the request body
     * It catches errors and handles the loading indicator
     * On success, it sets the state of output to be the output from the flask script
     * 
     * @param {String} myText 
     */
    handlePTSubmit = (myText) => {
        this.setState({
            input_type: 'plaintext',
            loading: true,
            results: false,
        })
        console.log("Input: " + myText);
        axios.post("/result", {myText})
            .then(res => {
                this.setState({ output: res.data })
                this.setState({ loading: false })
                this.setState({results: true})
                console.log(res.data);
            }).catch(err => {
                console.log(err);
                this.setState({ error: err.message });
                this.setState({ loading: false })
            });
    }
    /**
     * This method handles the url submit
     * It makes a POST request using the inputted url as the request body
     * It catches errors and handles the loading indicator
     * On success, it sets the state of output to be the output from the flask script
     * 
     * @param {String} input_url 
     */
    handleURLSubmit = (input_url) => {
        this.setState({
            input_type: 'url',
            loading: true,
            results: false,
        })
        console.log("Input: " + input_url);
        axios.post("/scrape", {input_url})
            .then(res => {
                this.setState({ output: res.data })
                this.setState({ loading: false })
                this.setState({results: true})
                console.log(res.data);
            }).catch(err => {
                console.log(err);
                this.setState({ error: err.message });
                this.setState({ loading: false })
            });
    }
    render() {
        return (
            <div>
                <Header />
                <InputWrapper 
                    handleURLSubmit={this.handleURLSubmit}
                    handlePTSubmit={this.handlePTSubmit}
                    loading={this.state.loading}
                    error={this.state.error}
                />
                <Results 
                    results={this.state.output}
                    is_populated={this.state.results}
                    input_type={this.state.input_type}
                />
            </div>
        )
    }
}