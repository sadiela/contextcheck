import React, {Component} from 'react';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';
import whatPOS from './GetPOS';
import axios from 'axios';

export default class TextPane extends Component {
    /**
     * This takes in the sentence level results and returns the words
     * If the word scored over the threshold, it will be red
     * The tooltip for red words includes the score of the word and the part of speech
     * Additionally, if it also has a word type, it will display that in the tooltip
     * 
     * @param {*} sentence 
     */
    getWords = (sentence) => {
        return(
            sentence.words.map(word => {
                const score = Math.round(word[1] * 100) / 100;
                const threshold = parseFloat(this.props.threshold);
                if(word[1] > threshold){
                    const part_of_speech = whatPOS(word[2]);
                    
                        return (
                                <OverlayTrigger
                                    key={word}
                                    placement='top'
                                    overlay={
                                        <Tooltip id={`tooltip-$word[0]`}>
                                            <p>Score: <strong>{score}</strong></p>
                                            <p>Part of Speech: <strong>{part_of_speech}</strong></p>
                                            <p>Word Type: <strong>{word[3]}</strong></p>
                                        </Tooltip>
                                    }                        
                                >
                                <span style={{color: 'red'}} className='word-level'>{word[0]}</span>
                                </OverlayTrigger>
                            )
                } else { // Case for when the word did not score higher than the threshold
                    return (
                        <span className='word-level'>{word[0]}</span>
                    )
                }
            })
        )
    }
    getSentences = (sentence) => {
            return(
                <span key={sentence.average} className='sentence-word-wrapper'>{this.getWords(sentence)}</span>
            )
    }
    render() {
        return(
            this.props.text.map(sentence => {
                return(
                    <div className='sentence-list-wrapper'>
                        {this.getSentences(sentence)}
                    </div>
                )
            })
        )
    }
}