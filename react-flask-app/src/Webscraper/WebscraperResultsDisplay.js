import React, { Component } from 'react';

export default class WebscraperResultsDisplay extends Component {
    constructor(props){
        super(props);
        this.state = {
            sorted: false,
        }
        this.handleSort = this.handleSort.bind(this);
    }
    getArticleContent() {
        if (this.props.output.title !== ""){
            var sentences = this.props.output.bias_results.sentence_results;
            if(this.state.sorted === true){
                sentences = this.props.output.bias_results.sentence_results.sort((a, b) => (a.bias_score > b.bias_score)? -1 : 1);
            }
            console.log(sentences);
            return sentences.map((sentence) => {
                return (
                    <div className='individual-sentence-wrapper'>
                        <p name='average'>{Math.round(sentence.bias_score * 100)/100}</p>
                        <p name='most-biased'>{sentence.max_biased_word}</p>
                    </div>
                );
            });
        } else {
            return (
                <div></div>
            )
        }
    }
    handleSort(){
        this.setState({sorted: !this.state.sorted});
    }
    render() {
        return(
            <div>
                <ul className='webscraper-results-list'>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='author'>Author</label>
                        <li name='author' className='webscraper-result'>{this.props.output.author}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='article-score'>Article Score</label>
                        <li name='article-score' className='webscraper-result'>{Math.round(this.props.output.bias_results.article_score * 100)/100}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='runtime'>Runtime</label>
                        <li name='runtime' className='webscraper-result'>{this.props.output.bias_results.runtime.substr(0,5)}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='related-articles'>Related Articles (Middle)</label>
                        <li name='related-articles' className='webscraper-result'>{this.props.output.related.middle}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='title'>Title</label>
                        <li name='title' className='webscraper-result'>{this.props.output.title}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='article'>Article Content</label>
                        <ul className='webscraper-article-content-list' name='article'>
                            <button onClick={this.handleSort}>Sort By Sentence Score</button>
                            <span className='sentences-header'>
                                <p>Sentence Score</p>
                                <p>Most Biased Word</p>
                            </span>
                            {this.getArticleContent()}
                        </ul>
                    </span>
                </ul>
            </div>
        )
    }
}