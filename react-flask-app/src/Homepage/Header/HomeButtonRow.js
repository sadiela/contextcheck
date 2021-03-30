import React, {Component} from 'react';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import logo from './1truecontext.png';

export default class HomeButtonRow extends Component {
    render() {
        return(
            <>
            <Navbar bg="dark" variant="dark">
                <a href="/"><img style={{height:'7.5vh', margin:'auto'}} src={logo} alt='ContextCheck'/></a>
                <Nav className="mr-auto">
                    <Nav.Link href="/">Bias Detector</Nav.Link>
                    <Nav.Link href="/about-us">About Us</Nav.Link>
                    <Nav.Link href="/deets">The Deets</Nav.Link>
                </Nav>
            </Navbar>
            </>
        )
    }
}