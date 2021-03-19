import React, {Component} from 'react';
import BiasIndicator from './BiasIndicator';
import Jubmotron from 'react-bootstrap/Jumbotron';
import TextPane from './TextPane';
import MetaWrapper from './MetaWrapper';

export default class ResultsWrapper extends Component {
    render() {
        if(this.props.is_populated && this.props.input_type === 'plaintext'){
            return(
                <Jubmotron>
                    <BiasIndicator 
                        bias_score={Math.round(this.props.results.article_score * 100) / 100}
                        runtime={this.props.results.runtime.slice(0,5)}
                    />
                    <TextPane
                        text={this.props.results.sentence_results}
                        threshold='0.7'
                    />
                </Jubmotron>
            )
        } else if (this.props.is_populated && this.props.input_type === 'url') {
            return(
                <Jubmotron>
                    <BiasIndicator 
                        bias_score={Math.round(this.props.results.bias_results.article_score * 100) / 100}
                        runtime={this.props.results.bias_results.runtime.slice(0,5)}
                    />
                    <MetaWrapper 
                        author={this.props.results.author}
                        related={this.props.results.related}
                        title={this.props.results.title}
                        date={this.props.results.date}
                    />
                    <TextPane
                        text={this.props.results.bias_results.sentence_results}
                        threshold='6'
                    />
                </Jubmotron>
            )
        } else {
            return(<div></div>)
        }
    }
}