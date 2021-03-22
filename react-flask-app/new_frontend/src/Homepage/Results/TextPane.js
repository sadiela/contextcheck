import React, {Component} from 'react';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';

export default class TextPane extends Component {
    getWords = (sentence) => {
        return(
            sentence.words.map(word => {
                const score = Math.round(word[1] * 100) / 100;
                const threshold = parseFloat(this.props.threshold);
                if(word[1] > threshold){
                    return (
                        <OverlayTrigger
                            key={word}
                            placement='top'
                            overlay={
                                <Tooltip id={`tooltip-$word[0]`}>
                                    Score: <strong>{score}</strong>
                                </Tooltip>
                            }                        
                        >
                        <span style={{color: 'red'}} className='word-level'>{word[0]}</span>
                        </OverlayTrigger>
                    )
                } else {
                    return (
                        <span className='word-level'>{word[0]}</span>
                    )
                }
            })
        )
    }
    getSentences = (sentence) => {
            return(
                <span key={sentence.average} className='sentence-word-wrapper' key={sentence.average}>{this.getWords(sentence)}</span>
            )
    }
    render() {
        return(
            this.props.text.map(sentence => {
                return(
                    <div className='sentence-list-wrapper'>
                        <ul className='sentences-list'>{this.getSentences(sentence)}</ul>
                    </div>
                )
            })
        )
    }
}