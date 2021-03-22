import React, {Component} from 'react';
import BiasIndicator from './BiasIndicator';
import Jubmotron from 'react-bootstrap/Jumbotron';
import Badge from 'react-bootstrap/Badge';
import TextPane from './TextPane';
import MetaWrapper from './MetaWrapper';
import RelatedArticles from './RelatedArticles';

export default class ResultsWrapper extends Component {
    render() {
        if(this.props.is_populated && this.props.input_type === 'plaintext'){
            return(
                    <div className='result-wrapper'>
                        <BiasIndicator 
                            bias_score={Math.round(this.props.results.article_score * 100) / 100}
                            runtime={this.props.results.runtime.slice(0,5)}
                        />
                        <h3><strong>Sentences</strong></h3>
                        <h6>Words in red may be biased, hover over them to see why.</h6>
                        <TextPane
                            text={this.props.results.sentence_results}
                            threshold='0.7'
                        />
                    </div>
            )
        } else if (this.props.is_populated && this.props.input_type === 'url') {
            const score = Math.round(this.props.results.bias_results.article_score * 100) / 100;
            let bias = '';
            let variant = '';
            if (score >= 7){bias = 'is probably biased'; variant='danger'}
            else if (score >=5){bias = 'could be biased'; variant='warning'}
            else if (score >= 3){bias = 'could be unbiased'; variant='primary'}
            else {bias = 'is probably unbiased'; variant='success'}
            return(
                <div className='result-wrapper'>
                    <h1>Results</h1>
                    <h4>ContextCheck thinks this article <Badge variant={variant}>{bias}</Badge></h4>
                    <p className='border'></p>
                    <div>
                        <BiasIndicator 
                            bias_score={score}
                            runtime={this.props.results.bias_results.runtime.slice(0,5)}
                        />
                        <MetaWrapper 
                            author={this.props.results.author}
                            title={this.props.results.title}
                            date={this.props.results.date}
                        />
                        <RelatedArticles
                            related={this.props.results.related}
                        />
                        <h3><strong>Sentences</strong></h3>
                        <h6>Words in red may be biased, hover over them to see why.</h6>
                        <TextPane
                            text={this.props.results.bias_results.sentence_results}
                            threshold='6'
                        />
                    </div>
                </div>
            )
        } else {
            return(<div></div>)
        }
    }
}