import React, { useState } from 'react';
import { Nav, Modal } from 'react-bootstrap';

function ReferenceModal() {
    const [show, setShow] = useState(false);

    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);

    return (
      <>
        <Nav.Link href="#" onClick={handleShow}>Reference</Nav.Link>
        <Modal show={show} onHide={handleClose}>
          <Modal.Header closeButton>
            <Modal.Title>Reference</Modal.Title>
          </Modal.Header>
          <Modal.Body>Reference Text</Modal.Body>
        </Modal>
      </>
    );
  }

export default ReferenceModal;