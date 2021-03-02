import React, { Component } from 'react';
import axios from 'axios';
import Form from './Form';
import PTResultsDisplay from './PTResultsDisplay';

export default class PTWrapper extends Component {
    constructor(props){
        super(props);
        this.state = {
            output: [{
                article_score: 0,
                runtime: "",
                sentence_results: [],
            }]
        }
    }
    handleSubmit = myText => {
        console.log("Input: " + myText);
        axios.post("/result", {myText})
            .then(res => {
                this.setState({ output: [...this.state.output, res.data] })
                console.log(res.data);
            });
    }
    render() {
        console.log("This output: ", this.state.output)
        if (this.state.output[1]){
            return (
                <div>
                    <Form 
                        handleSubmit={this.handleSubmit}
                    />
                    <PTResultsDisplay 
                        output={this.state.output}
                    />
                </div>
            )
        } else {
            return (
                <div>
                    <Form 
                        handleSubmit={this.handleSubmit}
                    />
                </div>
            )
        }
    }
}