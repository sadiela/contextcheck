import React, {Component} from 'react';
import HowToCard from './HowToCard';

export default class HowToWrapper extends Component {
    render() {
        return (
            <span className='how-to-use-steps'>
                <HowToCard 
                    step="Step One"
                    subtitle="User Input"
                    body="You can enter an article link or your own text."
                />
                <HowToCard 
                    step="Step Two"
                    subtitle="Analyze Results"
                    body="After hitting submit, wait for your results to load then see what we discovered."
                />
                <HowToCard 
                    step="Step Three"
                    subtitle="Learn More"
                    body="ContextCheck is meant to be a tool for you to come up with your own conclusion. Once you read through the results, check out the Learn More section."
                />
            </span>
        )
    }
}
