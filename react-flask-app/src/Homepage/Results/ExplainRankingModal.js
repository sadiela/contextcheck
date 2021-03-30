import React, { useState } from 'react';
import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import Alert from 'react-bootstrap/Alert';

export default function ExplainRankingModal() {
    const [show, setShow] = useState(false);
    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true)
        return (
            <>
                <Button variant='info' onClick={handleShow}>
                    Explain This Ranking
                </Button>

                <Modal show={show} onHide={handleClose}>
                    <Modal.Header closeButton>
                        <Modal.Title>What does this ranking mean?</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <h5>There are 4 rankings that we may hand out.</h5>
                        <p>Remember, these are just a tool to help you interpret your own understanding of the article's bias.</p>
                        <div className='explain-rank-wrap'>
                            <Alert variant='success'>Contains very little to no biased language</Alert>
                            <p>Our algorithm found barely any biased language! This reporting seems pretty objective.</p>
                        </div>
                        <div className='explain-rank-wrap'>
                            <Alert variant='info'>Contains some biased language</Alert>
                            <p>Our algorithm found some biased language. This reporting is probably objective, but look out for the highlighted words.</p>
                        </div>
                        <div className='explain-rank-wrap'>
                            <Alert variant='warning'>Contains fair amount of biased language</Alert>
                            <p>Our algorithm found a fair amount of biased language. This reporting might not be objective. Review the red words to see more.</p>
                        </div>
                        <div className='explain-rank-wrap'>
                            <Alert variant='danger'>Contains a large amount of biased language</Alert>
                            <p>Danger zone. Our algorithm found highly biased words in many sentences. This reporting is probably not objective. Make sure to review the red words. Also check the source tags (it may be an opinion piece)</p>
                        </div>
                    </Modal.Body>
                    <Modal.Footer>
                        <Button variant='secondary' onClick={handleClose}>
                            Close
                        </Button>
                    </Modal.Footer>
                </Modal>
            </>
        )
}