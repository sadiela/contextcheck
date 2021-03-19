import React, {Component} from 'react';
import Badge from 'react-bootstrap/Badge';

export default class BiasIndicator extends Component {
    render() {
            return (
                <div>
                    <h2>
                        Bias Score: <Badge variant="info">{this.props.bias_score}</Badge>
                    </h2>
                    <h3>
                        Runtime: <Badge variant="info">{this.props.runtime} seconds</Badge>
                    </h3>
                </div>
            )
    }
}