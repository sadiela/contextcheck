import React, {Component} from 'react';

export default class HowToCard extends Component {
    constructor(props){
        super(props);
        this.state = {
            step: this.props.step,
            subtitle: this.props.subtitle,
            body: this.props.body
        }
    }
    render() {
        return (
            <div>
                <span className='card-how-to'>
                    <h4 className='step-title'>{this.state.subtitle}</h4>
                    <p>{this.state.body}</p>
                </span>
                <br />
            </div>
            
        )
    }
}