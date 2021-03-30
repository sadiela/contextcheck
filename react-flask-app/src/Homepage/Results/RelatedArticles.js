import React, { Component } from 'react';
import Table from 'react-bootstrap/Table';

export default class Related extends Component {
    getTableBody() {
        return this.props.related.map(related_obj => {
            if(related_obj.Headline !== ""){
                return(
                    <tr key={related_obj.Source}>
                        <td>{related_obj.Source}</td>
                        <td><a rel="noreferrer" href={related_obj.URL} target="_blank">{related_obj.Headline}</a></td>
                    </tr>
                )
            } else {
                return(<></>)
            }
        });
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