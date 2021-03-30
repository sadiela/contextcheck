import React, { Component } from 'react';
import Header from '../Homepage/Header/HomeButtonRow';

export default class AboutUs extends Component {
    render() {
        return(
            <>
                <Header />
                <span className='full-deets-wrapper'>
                    <h1 className='the-question'>
                    </h1>
                    <span className='the-answer-body'>
                    </span>
                </span>
            </>
        )
    }
}