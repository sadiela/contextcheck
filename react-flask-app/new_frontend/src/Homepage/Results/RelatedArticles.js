import React, { Component } from 'react';
import Table from 'react-bootstrap/Table';

export default class Related extends Component {
    render() {
        return(
            <Table striped bordered hover>
            <thead>
                <tr>
                    <th>Related Article Source</th>
                    <th>Related Article Title</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Article Source</td>
                    <td><a href={this.props.related[0]}>Link</a></td>
                </tr>
                <tr>
                    <td>Article Source</td>
                    <td><a href={this.props.related[0]}>Link</a></td>
                </tr>
                <tr>
                    <td>Article Source</td>
                    <td><a href={this.props.related[0]}>Link</a></td>
                </tr>
                <tr>
                    <td>Article Source</td>
                    <td><a href={this.props.related[0]}>Link</a></td>
                </tr>
                <tr>
                    <td>Article Source</td>
                    <td><a href={this.props.related[0]}>Link</a></td>
                </tr>
            </tbody>
        </Table>
        )
    }
}