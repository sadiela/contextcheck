import React, {Component} from 'react';
import Table from 'react-bootstrap/Table';

export default class MetaWrapper extends Component {
    render() {
        return (
            <div className='meta-wrapper'>
                <Table striped bordered hover>
                    <thead>
                        <tr>
                            <th>Title</th>
                            <th>Author</th>
                            <th>Date</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>{this.props.title}</td>
                            <td>{this.props.author}</td>
                            <td>{this.props.date}</td>
                        </tr>
                    </tbody>
                </Table>
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
            </div>
        )
    }
}