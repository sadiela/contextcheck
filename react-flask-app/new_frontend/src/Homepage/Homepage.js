import React, { Component } from 'react';
import Header from './Header/HeaderWrapper';
import IOWrapper from './IOWrap/IOWrapper';
import Results from './Results/ResultsWrapper';
import axios from 'axios';

export default class Homepage extends Component {
    constructor(props){
        super(props);
        this.state = {
            loading: false,
            output: {
                author: [],
                bias_results: {},
                date: "",
                feedText: "",
                related: {},
                title: ""
            },
            results: false,
            input_type: '',
            error: '',
        }
        this.handlePTSubmit = this.handlePTSubmit.bind(this);
        this.handleURLSubmit = this.handleURLSubmit.bind(this);
    }
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
                this.setState({ error: err });
                this.setState({ loading: false })
            });
    }
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
                this.setState({ error: err });
                this.setState({ loading: false })
            });
    }
    render() {
        return (
            <div>
                <Header />
                <IOWrapper 
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