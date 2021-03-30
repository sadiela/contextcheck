import React, {Component} from 'react';
import HowToCard from './HowToCard';

export default class HowToWrapper extends Component {
    render() {
        return (
            <span className='how-to-use-steps'>
                <HowToCard 
                    subtitle="1: Input Text / URL"
                    body="You can enter an article link or your own text."
                />
                <HowToCard 
                    subtitle="2: Analyze Results"
                    body="After hitting submit, wait for your results to load then see what we discovered."
                />
                <HowToCard 
                    subtitle="3: Think"
                    body="Use ContextCheck as a tool for you to complement your own bias-interpretation. Check The Deets / About Us to learn more!"
                />
            </span>
        )
    }
}
