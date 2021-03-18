import React, {Component} from 'react';
import Card from 'react-bootstrap/Card';

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
                <Card style={{ width: '18rem', margin: 'auto', height: '13rem' }} bg='info'>
                    <Card.Body>
                        <Card.Title style={{ color: '#FFFFFF' }}><strong>{this.state.step}</strong></Card.Title>
                        <Card.Subtitle style={{ color: '#FFFFFF' }}>{this.state.subtitle}</Card.Subtitle>
                        <Card.Text>{this.state.body}</Card.Text>
                    </Card.Body>
                </Card>
                <br />
            </div>
            
        )
    }
}