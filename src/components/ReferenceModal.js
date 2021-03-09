import React, { useState } from 'react';
import { Nav, Modal } from 'react-bootstrap';

function ReferenceModal() {
    const [show, setShow] = useState(false);

    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);

    return (
      <>
        <Nav.Link style={{display: 'inline', padding: '0.5rem'}} href="#" onClick={handleShow}>Reference</Nav.Link>
        <Modal show={show} onHide={handleClose}>
          <Modal.Header closeButton>
            <Modal.Title>Reference</Modal.Title>
          </Modal.Header>
          <Modal.Body>Coming soon!</Modal.Body>
        </Modal>
      </>
    );
  }

export default ReferenceModal;