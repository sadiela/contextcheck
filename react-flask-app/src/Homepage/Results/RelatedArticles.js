import React, { Component } from 'react';
import Table from 'react-bootstrap/Table';

export default class Related extends Component {
    getTableBody() {
        return this.props.related.map(related_obj => {
            if(related_obj.Headline !== ""){
                return(
                    <tr>
                        <td>{related_obj.Source}</td>
                        <td><a href={related_obj.URL}>{related_obj.Headline}</a></td>
                    </tr>
                )
            }
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
                {this.getTableBody()}
            </tbody>
        </Table>
        )
    }
}