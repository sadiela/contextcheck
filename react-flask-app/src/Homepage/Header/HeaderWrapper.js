import React, {Component} from 'react';
import ButtonRow from './HomeButtonRow';
import HowTo from './HowToWrapper';

export default class HeaderWrapper extends Component {
    render() {
        return(
            <div>
                <ButtonRow />
                <hr className='divider'/>
                <HowTo />
            </div>
        )
    }
}