import React, {Component} from 'react';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';

export default class HomeButtonRow extends Component {
    render() {
        return(
            <Navbar bg="dark" variant="dark">
                <Navbar.Brand href="/">ContextCheck</Navbar.Brand>
                <Nav className="mr-auto">
                    <Nav.Link href="/">Bias Detector</Nav.Link>
                    <Nav.Link href="/about-us">About Us</Nav.Link>
                    <Nav.Link href="/deets">The Deets</Nav.Link>
                </Nav>
            </Navbar>
        )
    }
}