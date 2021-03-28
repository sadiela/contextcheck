import React, { Component } from 'react';
import './Form.css';

export default class PTResultsDisplay extends Component {
    constructor(props){
        super(props);
        this.state = {
            output: this.props.output
        }
    }
    getWords = sentence => {
        console.log(sentence);
        return sentence.map(word => {
            if (word[0] === '.'){
                return (<div></div>)
            }
            return (
                <p className='single-word'
                    style={{background: (word[1] > 0.5) ? "rgb(255, 208, 208)" : "rgb(208, 255, 210)"}}
                >{word[0]}</p>
            )
        })
    }
    getSentences = sentences => {
        return sentences.map(sentence => {
            return(
                <div className='single-sentence-wrapper'>
                    <p><strong>Score: </strong>{Math.round(sentence.bias_score*100)/100}</p>
                    <span className='single-sentence'>
                        {this.getWords(sentence.words)}
                    </span>
                </div>
            )
        })
    }
    getAResult = () => {
        return this.props.output.map (oneinput => {
            if (oneinput.article_score === 0){
                return (<div></div>)
            }
            return (
            <div className='result-wrapp'>
                <span className='pt-header-score-runtime'>
                    <p><strong>Bias: </strong>{Math.round(oneinput.article_score*100)/100}</p>
                    <p><strong>Runtime: </strong>{oneinput.runtime.substr(0,4)} seconds</p>
                </span>
                {this.getSentences(oneinput.sentence_results)}
            </div>
        )});
    }
    render() {
        return (
            <div>
                {this.getAResult()}
            </div>
        )
    }
}