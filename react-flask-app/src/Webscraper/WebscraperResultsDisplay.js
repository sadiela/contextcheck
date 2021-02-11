import React, { Component } from 'react';

export default class WebscraperResultsDisplay extends Component {
    getArticleContent() {
        const sentences = this.props.output.bias_results.sentence_results;
        console.log(sentences);
        return sentences.map((sentence) => {
            return (
                <div className='individual-sentence-wrapper'>
                    <p name='average'>{sentence.bias_score}</p>
                    <p name='most-biased'>{sentence.max_biased_word}</p>
                </div>
            );
        });
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
                        <li name='article-score' className='webscraper-result'>{this.props.output.bias_results.article_score}</li>
                    </span>
                    <span className='webscraper-result-wrapper'>
                        <label className='webscraper-result-label' htmlFor='runtime'>Runtime</label>
                        <li name='runtime' className='webscraper-result'>{this.props.output.bias_results.runtime}</li>
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