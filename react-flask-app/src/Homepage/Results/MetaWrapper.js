import React, {Component} from 'react';
import Table from 'react-bootstrap/Table';

export default class MetaWrapper extends Component {
    render() {
        return (
            <div className='meta-wrapper'>
                <Table striped bordered hover>
                    <thead>
                        <tr>
                            <th>Article Title</th>
                            <th>Source Type</th>
                            <th>Author</th>
                            <th>Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{this.props.title}</td>
                            <td>{this.props.source_type}</td>
                            <td>{this.props.author}</td>
                            <td>{this.props.date}</td>
                        </tr>
                    </tbody>
                </Table>
            </div>
        )
    }
}