import React, { Component } from 'react';
import axios from 'axios';
import ScrapeInput from './ScrapeInput';
import WebscraperResultsDisplay from './WebscraperResultsDisplay';

export default class ScraperWrapper extends Component {
    constructor(props){
        super(props);
        this.state = {
            output: {
                author: [],
                bias_results: {},
                date: "",
                feedText: "",
                related: {},
                title: ""
            }
        }
    }

    handleSubmit = input_url => {
        console.log("Input: " + input_url);
        axios.post("/scrape", {input_url})
            .then(res => {
                this.setState({ output: res.data })
                console.log(res.data);
            });
    }
    render() {
        return(
            <div>
                <ScrapeInput 
                    handleSubmit={this.handleSubmit}
                />
                <WebscraperResultsDisplay 
                    output={this.state.output}
                />
            </div>
        )
    }

}