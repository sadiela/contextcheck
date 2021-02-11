import React, { Component } from 'react';

export default class WebscraperResultsDisplay extends Component {
    constructor(props){
        super(props);
        this.state = {
            sorted: false,
            show_sentence: false,
        }
        this.handleSort = this.handleSort.bind(this);
    }
    sentenceClickHandler = sentence => {
        this.setState({show_sentence: !this.state.show_sentence})
    }
    showSentence = sentence => {
        if(this.state.show_sentence){
            return sentence.words.map ((word) => {
                return (
                    <p>{word[0].replace('##', '')} {word[2]}</p>
                )
            })
        }
        else {
            return(
                <div></div>
            )
        }
    }
    getArticleContent() {
        if (this.props.output.title !== ""){
            var sentences = this.props.output.bias_results.sentence_results;
            if(this.state.sorted === true){
                sentences = this.props.output.bias_results.sentence_results.sort((a, b) => (a.bias_score > b.bias_score)? -1 : 1);
            } else {
                sentences = this.props.output.bias_results.sentence_results.sort((a,b) => (a.order > b.order) ? 1 : -1)
            }
            console.log(sentences);
            return sentences.map((sentence) => {
                return (
                    <div 
                        className='individual-sentence-wrapper'
                        onClick={this.sentenceClickHandler}
                    >
                        <p style={{color: (sentence.bias_score > 6) ? "red" : "green"}} name='average'>{Math.round(sentence.bias_score * 100)/100}</p>
                        <p name='most-biased'>{sentence.max_biased_word.replace('##', '')}</p>
                        {this.showSentence(sentence)}
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
                        <li style={{color: (this.props.output.bias_results.article_score > 6) ? "red" : "green"}} name='article-score' className='webscraper-result'>{Math.round(this.props.output.bias_results.article_score * 100)/100}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='runtime'>Runtime</label>
                        <li name='runtime' className='webscraper-result'>{this.props.output.bias_results.runtime.substr(0,5)} seconds</li>
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
                            <button onClick={this.handleSort}>Toggle Sort</button>
                            <p>Currently sorting by <strong>{(this.state.sorted) ? "bias score" : "order of appearance"}</strong></p>
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