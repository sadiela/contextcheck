import React, { Component } from 'react';

export default class WebscraperResultsDisplay extends Component {
    render() {
        return(
            <div>
                <ul>
                    <li>Author(s): {this.props.output.author}</li>
                    <li>Score: {this.props.output.bias_results.article_score}</li>
                    <li>Runtime: {this.props.output.bias_results.runtime}</li>
                    <li>Related Articles: Middle: {this.props.output.related.middle}</li>
                    <li>Title: {this.props.output.title}</li>
                </ul>
            </div>
        )
    }
}