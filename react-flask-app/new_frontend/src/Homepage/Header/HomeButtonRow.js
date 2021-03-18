import React, {Component} from 'react';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';

export default class HomeButtonRow extends Component {
    render() {
        return(
            <Navbar bg="dark" variant="dark">
                <Navbar.Brand href="#home">ContextCheck</Navbar.Brand>
                <Nav className="mr-auto">
                    <Nav.Link href="#home">Bias Detector</Nav.Link>
                    <Nav.Link href="#features">About Us</Nav.Link>
                    <Nav.Link href="#pricing">The Deets</Nav.Link>
                </Nav>
            </Navbar>
        )
    }
}