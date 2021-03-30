import React, { Component } from 'react';
import Dropdown from 'react-bootstrap/Dropdown';

const SUPPORTED_URLS = [
    "cnn.com",
    "foxnews.com",
    "huffpost.com",
    "slate.com",
    "reuters.com",
    "progressive.org",
    "politico.com",
    "theguardian.com",
    "apnews.com",
    "cbsnews.com",
    "cnbc.com",
]

export default class URLDropdown extends Component {
    getDropdownItems(){
        return SUPPORTED_URLS.map(url => {
            return(
                <Dropdown.Item>{url}</Dropdown.Item>
            )
        })
    }
    render(){
        return(
            <div className='supported-dropdown'>
                <Dropdown>
                    <Dropdown.Toggle variant="info" id="dropdown-basic">
                        See Supported URLs
                    </Dropdown.Toggle>
                    <Dropdown.Menu>
                        {this.getDropdownItems()}
                    </Dropdown.Menu>
                </Dropdown>
            </div>
        )
    }
}