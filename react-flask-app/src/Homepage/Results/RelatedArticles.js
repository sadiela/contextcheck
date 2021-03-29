import React, { Component } from 'react';
import Table from 'react-bootstrap/Table';

export default class Related extends Component {
    getTableBody() {
        return this.props.related.map(related_obj => {
            // If not empty return this, else return empty
            return(
                <tr>
                    <td>Article Source Goes Here</td>
                    <td>Related Article Link Goes Here (with title)</td>
                </tr>
            )
        })
    }
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