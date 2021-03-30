import React, {Component} from 'react';
import Badge from 'react-bootstrap/Badge';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';
import ExplainRankingModal from './ExplainRankingModal';

export default class BiasIndicator extends Component {
    // render() {
    //         return (
    //             <div>
    //                 <h2>
    //                     Bias Score: <Badge variant="info">{this.props.bias_score}</Badge>
    //                 </h2>
    //                 <h3>
    //                     Runtime: <Badge variant="info">{this.props.runtime} seconds</Badge>
    //                 </h3>
    //             </div>
    //         )
    // }
    render() {
        return (
            <OverlayTrigger
                key={this.props.bias_score}
                placement='right'
                overlay={
                    <Tooltip id={`tooltip-$this.props.bias_score`}>
                        <p>Score: <strong>{this.props.bias_score} / 10</strong></p>
                    </Tooltip>
                }                        
            >
                <div className='we-think'>
                    <h3>ContextCheck thinks this article <Badge variant={this.props.variant}>{this.props.end_sentence}</Badge></h3>
                    <h4>Runtime: {this.props.runtime} seconds</h4>
                    <ExplainRankingModal />
                </div>
            </OverlayTrigger>
        );
    }
}